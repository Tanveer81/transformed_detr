# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
import random
from pathlib import Path
import numpy as np
import torchvision
import torch
import albumentations as A
import torchvision.transforms.functional as F
from .copy_paste import copy_paste_class, CopyPaste
from pycocotools import mask as coco_mask
from PIL import Image
import os
SOA_THRESH = 50 * 50
SOA_PROB = 0.5
SOA_COPY_TIMES = 2
SOA_EPOCHS = 30
SOA_ONE_OBJECT = False
SOA_ALL_OBJECTS = False

min_keypoints_per_image = 10

def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)

def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True

    return False

@copy_paste_class
class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, mixed_augment, copy_paste_augment):
        super(CocoDetection, self).__init__(img_folder, ann_file, None, None, copy_paste_augment)
        # filter images without detection annotations
        ids = []
        for img_id in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
            anno = self.coco.loadAnns(ann_ids)
            if has_valid_annotation(anno):
                ids.append(img_id)
        self.ids = ids
        self.mixed_augment = mixed_augment
        self.prepare = ConvertCocoPolysToMask(True)

    def load_example(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        target = self.coco.loadAnns(ann_ids)

        path = self.coco.loadImgs(img_id)[0]['file_name']
        image = Image.open(os.path.join(self.root, path)).convert('RGB')

        image = np.array(image)

        masks = []
        bboxes = []

        h, w = image.shape[0], image.shape[1]
        target = {'image_id': img_id, 'annotations': target}
        target = self.prepare(h, w, target)
        for ix in range(len(target['boxes'])):
        # for ix, obj in enumerate(target):
            box = target['boxes'][ix].tolist()
            mask = target['masks'][ix].numpy()
            category_id = target['labels'][ix].item()
            box[0], box[2] = box[0] / w, box[2] / w
            box[1], box[3] = box[1] / h, box[3]/ h
            bboxes.append(box + [category_id] + [ix])
            masks.append(mask)
        # pack outputs into a dict
        output = {
            'image': image,
            'masks': masks,
            'bboxes': bboxes
        }
        return self.mixed_augment(**output), target

def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, h, w, target):

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return target

def RandomResize(img, **params):
    # Bbox are in normalized form [0,1]. So no need to resize bbox.
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    size = random.choice(scales)
    random_resize = A.Resize(size, size)
    return random_resize(image=img)['image']

def mixed_augmentation(image_set, image_size):
    normalize = A.Compose([A.Resize(image_size[0], image_size[1]),  # height first for this library
                           A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    if image_set == 'train':
        color_aug_list = [
            A.RandomBrightnessContrast(p=1), A.RandomBrightnessContrast(contrast_limit=0., p=1),
            A.RandomBrightnessContrast(brightness_limit=0., p=1), A.RGBShift(),
            A.HueSaturationValue(p=1),
            A.ChannelShuffle(p=1), A.CLAHE(p=1), A.RandomGamma(p=1), A.Blur(p=1), A.ToGray(p=1),
            A.ToSepia(p=1), ]
        # Return random augmentation with 0.7 probability
        random_color_aug = \
        random.choices([A.NoOp(), random.choice(color_aug_list)], weights=[0.3, 0.7])[0]

        detr_aug = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.OneOrOther(
                A.NoOp(),
                A.Compose([
                    A.Lambda(p=1, image=RandomResize),
                    A.RandomSizedBBoxSafeCrop(384, 600, p=1)
                ])
            )
        ])

        spacial_aug_list = [A.HorizontalFlip(p=1), A.Flip(p=1),  # vertical
                            A.Transpose(p=1), A.RandomRotate90(p=1), A.RandomSizedBBoxSafeCrop(384, 600, p=1),
                            A.LongestMaxSize(p=1), ]
        random_spacial_aug = random.choices([A.NoOp(), random.choice(spacial_aug_list), detr_aug], weights=[0.4, 0, 0.6])[0]

        # transform = A.Compose([random_spacial_aug, random_color_aug, normalize],
        #                       bbox_params=A.BboxParams(format='albumentations'))
        transform = A.Compose(normalize,
                              bbox_params=A.BboxParams(format='albumentations'))
        return transform

    elif image_set == 'val':
        return A.Compose(normalize,
                         bbox_params=A.BboxParams(format='albumentations'), )

    raise ValueError(f'unknown {image_set}')

def copy_paste_augmentation(image_set, image_size):
    def toTensor(img, **params):
        return F.to_tensor(img)

    if image_set == 'train':
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.OneOrOther(
                A.NoOp(),
                A.Compose([
                    A.Lambda(p=1, image=RandomResize),
                    A.RandomSizedBBoxSafeCrop(384, 600, p=1)
                ])
            ),
            A.RandomScale(scale_limit=(-0.9, 1), p=1),  # LargeScaleJitter from scale of 0.1 to 2
            A.PadIfNeeded(256, 256, border_mode=0),  # constant 0 border
            A.RandomCrop(256, 256),
            A.HorizontalFlip(p=0.5),
            CopyPaste(blend=True, sigma=1, pct_objects_paste=0.5, p=1),
            A.Lambda(p=1, image=toTensor),
        ], bbox_params=A.BboxParams(format="albumentations"))

    if image_set == 'val':
        return A.Compose(
            [A.Lambda(p=1, image=toTensor)]
        , bbox_params=A.BboxParams(format="albumentations"))

def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
    }
    img_folder, ann_file = PATHS[image_set]
    # Use transformer for ViT
    dataset = CocoDetection(img_folder, ann_file,
                            mixed_augment=mixed_augmentation(image_set, args.img_size),
                            copy_paste_augment=copy_paste_augmentation(image_set, args.img_size))
    return dataset
