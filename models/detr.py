# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import numpy as np
import torch
import os
import torch.nn.functional as F
from torch import nn
from torchvision.ops import nms
import matplotlib.pyplot as plt

from timm import create_model
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, patchify, unpatchify)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .transformer import build_transformer
from .pytorch_pretrained_vit.utils import load_pretrained_weights
from models.pytorch_pretrained_vit.model import ViT
# from models.pytorch_pretrained_vit.model import hierarchicalViT
from models.pytorch_pretrained_vit.configs import PRETRAINED_MODELS
from .pytorch_pretrained_vit.utils import non_strict_load_state_dict


class DETR(nn.Module):
    """ This is the DETR module that performs object detection """

    def __init__(self, backbone, transformer, num_classes, num_queries, imsize, datasize,
                 aux_loss=False, cls_token=False, distilled=False, deit=False, patch_vit=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.backbone = backbone
        self.init = True
        self.aux_loss = aux_loss
        self.imsize = imsize
        self.datasize = datasize
        self.distilled = distilled
        self.cls_token = cls_token
        self.deit = deit #todo remove , just for a wrokarind of clas token in fwd
        self.patch_vit = patch_vit
        if deit:# This if else condition is needed for VIT code compatibility. Can remove it when shieft to timm code totally
            self.backbone_dim = backbone.embed_dim
        else: # VIT
            self.backbone_dim = backbone.dim
        # If backbone and detr has different hidden dimension, we create projection for compatability
        if self.backbone_dim != transformer.d_model:
            self.hidden_dim_proj_src = nn.Linear(self.backbone_dim, transformer.d_model)
            self.hidden_dim_proj_pos = nn.Linear(self.backbone_dim, transformer.d_model)

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        if not isinstance(samples, torch.Tensor):
            samples = samples.tensors

        if self.patch_vit: # TODO hardcoded for 224x224 patch and 560x560 image
            patch_size = self.imsize
            step = (int(patch_size[0]/2),int(patch_size[1]/2))
            samples, n_patch = patchify(samples, patch_size, step)
        
        src, pos = self.backbone(samples)

        if self.deit:
            if self.distilled:
                cls_dist_token, pos_token = pos[:, :2, :], pos[:, 2:, :]
                src_token = src[:,2:,:]
            elif not self.cls_token:
                cls_token, pos_token = pos[:, 0:, :],pos[:, 1:, :]
                src_token = src[:, 1:, :]
                # pos = self.backbone.transformer.hour_glass(pos[:, 1:, :]) #todo @tanveer later fix for hierchy, mayb not needed
                # pos = torch.cat([token, pos], 1).contiguous()
        else:  #todo @tanver we dnt need to remove cls token inside vit, do it here and align else here
                pos_token = pos
                src_token = src
        #         pos = self.backbone.transformer.hour_glass(pos)
        # if self.backbone.position_embedding == "learned":
        #     pos = pos.expand(src.shape[0], pos.shape[1], pos.shape[2])

        mask = None
        # In case of ViT DETR transformer would not include encoder

        # If backbone and detr has different hidden dimension, we create projection for compatability
        if self.backbone_dim != self.transformer.d_model:
            src_token = self.hidden_dim_proj_src(src_token)
            pos_token = self.hidden_dim_proj_pos(pos_token)

        if self.patch_vit: # TODO hardcoded for 224x224 patch and 560x560 image
            patch_size = (int(patch_size[0]/16), int(patch_size[1]/16))
            step = (int(patch_size[0]/2),int(patch_size[1]/2))
            new_patchsize = (int(self.datasize[0]/16), int(self.datasize[1]/16))
            src_token = src_token.view(-1,n_patch[0],n_patch[1],patch_size[0],patch_size[1],self.transformer.d_model).permute(0,5,1,2,3,4)
            src_token = unpatchify(src_token, step)
            src_token = src_token.view(-1,self.transformer.d_model,new_patchsize[0]*new_patchsize[1]).permute(0,2,1)

            pos_token = pos_token.repeat(n_patch[0]*n_patch[1],1,1)
            pos_token = pos_token.view(-1,n_patch[0],n_patch[1],patch_size[0],patch_size[1],self.transformer.d_model).permute(0,5,1,2,3,4)
            pos_token = unpatchify(pos_token, step)
            pos_token = pos_token.view(-1,self.transformer.d_model,new_patchsize[0]*new_patchsize[1]).permute(0,2,1)

        hs = self.transformer(src_token, mask, self.query_embed.weight, pos_token)

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses, loss_type):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.loss_type = loss_type
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """

        def balanced_l1_loss(pred,
                             target,
                             beta=1.0,
                             alpha=0.5,
                             gamma=1.5,
                             reduction='mean'):
            """Calculate balanced L1 loss.

            Please see the `Libra R-CNN <https://arxiv.org/pdf/1904.02701.pdf>`_

            Args:
                pred (torch.Tensor): The prediction with shape (N, 4).
                target (torch.Tensor): The learning target of the prediction with
                    shape (N, 4).
                beta (float): The loss is a piecewise function of prediction and target
                    and ``beta`` serves as a threshold for the difference between the
                    prediction and target. Defaults to 1.0.
                alpha (float): The denominator ``alpha`` in the balanced L1 loss.
                    Defaults to 0.5.
                gamma (float): The ``gamma`` in the balanced L1 loss.
                    Defaults to 1.5.
                reduction (str, optional): The method that reduces the loss to a
                    scalar. Options are "none", "mean" and "sum".

            Returns:
                torch.Tensor: The calculated loss
            """
            assert beta > 0
            assert pred.size() == target.size() and target.numel() > 0

            diff = torch.abs(pred - target)
            b = np.e ** (gamma / alpha) - 1
            loss = torch.where(
                diff < beta, alpha / b *
                (b * diff + 1) * torch.log(b * diff / beta + 1) - alpha * diff,
                gamma * diff + gamma / b - alpha * beta)

            return loss

        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        if self.loss_type == 'l1':
            loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        elif self.loss_type == 'smooth_l1':
            # make a new copy of the src and target bboxes
            sqrt_src_boxes = src_boxes.detach().clone()
            sqrt_target_boxes = target_boxes.detach().clone()
            # sqrt only the height and width to emphasize small objects more
            sqrt_src_boxes[:, 2:4] = torch.sqrt(sqrt_src_boxes[:, 2:4])
            sqrt_target_boxes[:, 2:4] = torch.sqrt(sqrt_target_boxes[:, 2:4])
            loss_bbox = F.smooth_l1_loss(sqrt_src_boxes, sqrt_target_boxes, reduction='none')
        elif self.loss_type == 'balanced_l1':
            # make a new copy of the src and target bboxes
            sqrt_src_boxes = src_boxes.detach().clone()
            sqrt_target_boxes = target_boxes.detach().clone()
            # sqrt only the height and width to emphasize small objects more
            sqrt_src_boxes[:, 2:4] = torch.sqrt(sqrt_src_boxes[:, 2:4])
            sqrt_target_boxes[:, 2:4] = torch.sqrt(sqrt_target_boxes[:, 2:4])
            loss_bbox = balanced_l1_loss(sqrt_src_boxes, sqrt_target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float,
                                    device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self, use_nms=False):
        """ Create the criterion.
        Parameters:
            nms: non maximum supression for detected object
        """
        super().__init__()
        self.use_nms = use_nms

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        if self.use_nms:  # perform NMS
            results = []
            for s, l, b in zip(scores, labels, boxes):
                keep = nms(b, s, iou_threshold=0.5)
                results.append({'scores': s[keep], 'labels': l[keep], 'boxes': b[keep]})
        else:
            results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    num_classes = 20 if args.dataset_file != 'coco' else 91
    if args.dataset_file == "coco_panoptic":
        # for panoptic, we just add a num_classes that is large enough to hold
        # max_obj_id + 1, but the exact value doesn't really matter
        num_classes = 250
    device = torch.device(args.device)

    # if args.hierarchy:
    #     args.backbone_name = "ViT"
    #     backbone = hierarchicalViT(args)
    #     # trasformer d_model
    #     args.hidden_dim = PRETRAINED_MODELS[args.backbone]['config']['dim']

    if args.backbone == "ViT":
        backbone = ViT(args.pretrained_model, pretrained=args.pretrained_vit, pretrain_dir=args.pretrain_dir, detr_compatibility=True,
                       position_embedding=args.position_embedding, image_size=args.img_size, num_heads=args.backbone_nheads, num_layers=args.enc_layers,
                       include_class_token=args.include_class_token, skip_connection=args.skip_connection, hierarchy=args.hierarchy, pool=args.pool, deit='Deit' in args.backbone,)
        # Make detr d_model compatible with vit
        args.hidden_dim = PRETRAINED_MODELS[args.pretrained_model]['config']['dim']

    else:  # for deit
        backbone = create_model(args.pretrained_model,
                                    pretrained=False,
                                    num_classes=1000,
                                    drop_rate=args.dropout,
                                    drop_path_rate=args.drop_path,
                                    drop_block_rate=None,
                                    skip_connection = args.skip_connection,
                                    img_size= (args.img_height, args.img_width), #todo for variable im wdith and heoight
                                )
        # Make detr d_model compatible with deit
        # args.hidden_dim = backbone.embed_dim  # TODO: remove this line
        if args.pretrained_vit:
            pretrained_image_size = np.repeat(int(args.pretrained_model.split('_')[-1]), 2)  # model contains sq image size in the name ex. deit_base_patch16_384
            patch_size = backbone.patch_embed.patch_size
            load_pretrained_weights(
                backbone,
                weights_path=args.pretrain_dir,
                load_first_conv=True,
                resize_positional_embedding=args.img_size != tuple(pretrained_image_size),
                old_img=(pretrained_image_size[0] // patch_size[0], pretrained_image_size[1] // patch_size[1]),  # original vit/deit 384x384
                new_img=(args.img_size[0] // patch_size[0], args.img_size[1] // patch_size[1]),  # todo experiment with height and weight
                deit='Deit' in args.backbone,
                distilled='distilled' in args.pretrained_model
            )

    # If we want to load pretrained detr, we need to keep the dmodel = 256
    # Otherwise it will be fixed according to backbone spec above
    if os.path.exists(args.detr_pretrain_dir) > 0:
        args.hidden_dim = 256 # as pretrained detr dim was 256

    transformer = build_transformer(args)

    #TODO: Load weights here
    if os.path.exists(args.detr_pretrain_dir)>0:
        state_dict = torch.load(args.detr_pretrain_dir)

        for old_key, old_val in list(state_dict['model'].items()):
            if any(k in old_key for k in ['encoder', 'backbone']) : # delete unnecessary kwys with enoder,bconv backbone as on decoder weight is needed
                del state_dict['model'][old_key]
            else:
                del state_dict['model'][old_key]
                new_key = old_key.replace('transformer.', '')
                state_dict['model'][new_key] = old_val

        ret = non_strict_load_state_dict(transformer, state_dict['model'])

        print('Missing keys when loading pretrained weights: {}'.format(ret.missing_keys))
        print('Unexpected keys when loading pretrained weights: {}'.format(ret.unexpected_keys))
        print('Loaded Detr weight from %s'%args.detr_pretrain_dir)

    model = DETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        imsize=(args.img_height, args.img_width),
        datasize=(args.data_height, args.data_width),
        aux_loss=args.aux_loss,
        cls_token=args.include_class_token,
        distilled='distilled' in  args.pretrained_model,
        deit='Deit' in args.backbone,
        patch_vit=args.patch_vit,
    )

    if os.path.exists(args.detr_pretrain_dir) > 0:
        model.class_embed.weight.data.copy_(state_dict['model']['class_embed.weight'])
        model.class_embed.bias.data.copy_(state_dict['model']['class_embed.bias'])
        model.query_embed.weight.data.copy_(state_dict['model']['query_embed.weight'])
        model.bbox_embed.layers[0].weight.data.copy_(state_dict['model']['bbox_embed.layers.0.weight'])
        model.bbox_embed.layers[0].bias.data.copy_(state_dict['model']['bbox_embed.layers.0.bias'])
        model.bbox_embed.layers[1].weight.data.copy_(state_dict['model']['bbox_embed.layers.1.weight'])
        model.bbox_embed.layers[1].bias.data.copy_(state_dict['model']['bbox_embed.layers.1.bias'])
        model.bbox_embed.layers[2].weight.data.copy_(state_dict['model']['bbox_embed.layers.2.weight'])
        model.bbox_embed.layers[2].bias.data.copy_(state_dict['model']['bbox_embed.layers.2.bias'])

    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses, loss_type = args.loss_type)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors
