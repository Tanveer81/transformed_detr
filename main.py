# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model
from models.pytorch_pretrained_vit.configs import PRETRAINED_MODELS
from models.pytorch_pretrained_vit.utils import resize_positional_embedding_, maybe_print
from tensorboardX import SummaryWriter

torch.multiprocessing.set_sharing_strategy('file_system')
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
#import wandb
from argparse import Namespace


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)

    # Transformed Detection Args
    parser.add_argument('--experiment_name', default='train', type=str)
    parser.add_argument('--overfit_one_batch', default=False, action='store_true')
    parser.add_argument('--pretrained_vit', default=False, action='store_true')
    parser.add_argument('--pretrained_model', default='B_16_imagenet1k', type=str,
                        help="ViT pre-trained model type")
    parser.add_argument('--pretrain_dir', default='/mnt/data/hannan/.cache/torch/checkpoints',
                        help='path where to save, empty for no saving')
    parser.add_argument('--random_image_size', default=False, action='store_true')
    parser.add_argument('--img_width', default=384, type=int)
    parser.add_argument('--img_height', default=384, type=int)
    parser.add_argument('--vit_heads', default=12, type=int)
    parser.add_argument('--vit_layer', default=12, type=int)
    parser.add_argument('--include_class_token', default=False, action='store_true')
    parser.add_argument('--skip_connection', default=False, action='store_true')
    parser.add_argument('--hierarchy', default=False, action='store_true')
    parser.add_argument('--only_weight', action='store_true', help='used for coco trainined detector')
    parser.add_argument('--pool', default='max', type=str, choices=('max', 'avg'))
    parser.add_argument('--augment', default=False, action='store_true')
    parser.add_argument('--deit', default=False, action='store_true')
    parser.add_argument('--opt', default='AdamW', type=str, choices=('AdamW', 'SGD'))

    # Training
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--print_freq', default=500, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='1d', type=str,
                        choices=('1d', '2d', 'sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')


    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    return parser


def main(args):
    # wandb.login()
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)

    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],
                                                          find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if
                    "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if
                       "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    if args.opt=="AdamW":
        print("Train with AdamW")
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)
    else:
        print("Train with SGD")
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=8, factor=0.1,
                                   verbose=True, threshold=0.001, threshold_mode='abs', cooldown=1)

    dataset_train = build_dataset(image_set='train', args=args)

    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn,
                                 num_workers=args.num_workers)

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
            # If we want to transfer learn on new image size, we need to resize positional embedding of ViT
            if args.backbone in PRETRAINED_MODELS.keys():
                pretrained_image_size = PRETRAINED_MODELS[args.backbone]['image_size']
                if args.img_size != pretrained_image_size:
                    old_img = (pretrained_image_size[0] // model_without_ddp.backbone.fh, pretrained_image_size[1] // model_without_ddp.backbone.fw),
                    new_img = (model_without_ddp.backbone.gh, model_without_ddp.backbone.gw)
                    posemb = checkpoint['model']['backbone.positional_embedding.pos_embedding']
                    posemb_new = model_without_ddp.state_dict()['backbone.positional_embedding.pos_embedding']
                    checkpoint['model']['backbone.positional_embedding.pos_embedding'] = \
                    resize_positional_embedding_(posemb=posemb, posemb_new=posemb_new,
                                                    has_class_token=hasattr(model_without_ddp.backbone, 'class_token'),
                                                    gs_old=old_img[0], gs_new=new_img)
                    maybe_print('Resized positional embeddings from {} to {}'.format(
                        posemb.shape, posemb_new.shape), True)
            print('Resumming Model from:', args.resume)
            ret = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
            maybe_print('Missing keys when loading pretrained weights: {}'.format(ret.missing_keys), True)
            maybe_print('Unexpected keys when loading pretrained weights: {}'.format(ret.unexpected_keys), True)
            if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint  and not args.only_weight:
                print('Resumming Optimizer from:', args.resume)
                optimizer.load_state_dict(checkpoint['optimizer'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return

    # Create tensorboard writer
    writer = SummaryWriter(comment=args.experiment_name)

    # tell wandb to get started
    # if os.environ.get("RANK", "0") == "0":
    #     wandb.init(project=args.experiment_name, config=args, dir="../")

    print("Start training")
    # tell wandb to watch what the model gets up to: gradients, weights, and more!
    # if os.environ.get("RANK", "0") == "0":
    #     wandb.watch(model, criterion, log="all", log_freq=10)
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        print(f"epoch:{epoch}")
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm, args.overfit_one_batch, args.print_freq)

        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,
            args.overfit_one_batch
        )
        ap_box = coco_evaluator.coco_eval['bbox'].stats.tolist()[0]  # take AP Box for reducing lr
        lr_scheduler.step(ap_box)
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        map_keys = [
            'AP_IoU=0.50:0.95_all_maxDets=100',
            'AP_IoU=0.50_all_maxDets=100',
            'AP_IoU=0.75_all_maxDets=100',
            'AP_IoU=0.50:0.95_small_maxDets=100',
            'AP_IoU=0.50:0.95_medium_maxDets=100',
            'AP_IoU=0.50:0.95_large_maxDets=100',
            'AR_IoU=0.50:0.95_all_maxDets=1',
            'AR_IoU=0.50:0.95_all_maxDets=10',
            'AR_IoU=0.50:0.95_all_maxDets=100',
            'AR_IoU=0.50:0.95_small_maxDets=100',
            'AR_IoU=0.50:0.95_medium_maxDets=100',
            'AR_IoU=0.50:0.95_large_maxDets=100',
        ]

        # if os.environ.get("RANK", "0") == "0":
        for metric, iou in zip(map_keys, coco_evaluator.coco_eval['bbox'].stats.tolist()):
            writer.add_scalar(metric, iou, epoch)
            # wandb.log({metric: iou, 'epoch': epoch})

        for k, v in train_stats.items():
            if isinstance(v, float):
                writer.add_scalar(f'train_{k}', v, epoch)
                # wandb.log({f'train_{k}': v, 'epoch': epoch})
        for k, v in test_stats.items():
            if isinstance(v, float):
                writer.add_scalar(f'test_{k}', v, epoch)
                # wandb.log({f'test_{k}': v, 'epoch': epoch})

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    time_per_batch = total_time / args.print_freq
    print('Training time {}'.format(total_time_str))
    print("\n{:.1f}mmaster_addr/epoch".format(len(data_loader_train) * time_per_batch / 60))

    writer.close()


def inference(args=None, resume='', skip_connection=False, img_width=384, img_height=384):
    if args==None:
        parser = argparse.ArgumentParser('DETR training and evaluation script',
                                         parents=[get_args_parser()])
        args = vars(parser.parse_args([]))
        args = Namespace(**args)
        args.img_width=img_width
        args.img_height=img_height
        args.img_size = (args.img_height, args.img_width)
        print(args)
        if not args.output_dir:  # create output dir as per experiment name in exp folder
            args.output_dir = './exp/' + args.experiment_name
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    args.resume = resume
    args.skip_connection = skip_connection
    args.pretrained_vit = True
    args.pretrain_dir = '/nfs/data3/koner/data/checkpoints/vit_detr/'
    args.backbone = 'B_16_imagenet1k'

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)

    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],
                                                          find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if
                    "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if
                       "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')

    for n in checkpoint['model'].copy().keys():
        if n.startswith(('transformer.encoder')):  # or n.startswith('freq_bias')
            del (checkpoint['model'][n])

    model_without_ddp.load_state_dict(checkpoint['model'])
    if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script',
                                     parents=[get_args_parser()])
    args = parser.parse_args()
    args.img_size = (args.img_height, args.img_width)
    print(args)
    if not args.output_dir:  # create output dir as per experiment name in exp folder
        args.output_dir = './exp/' + args.experiment_name
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
    # model = inference(args, resume = '/nfs/data3/koner/data/checkpoints/vit_detr/exp/skip_connection_wdNorm/checkpoint.pth', skip_connection=True)
    # model = inference(args, resume='/mnt/data/hannan/deit/deit_base_patch16_224-b5f2ef4d.pth', skip_connection=False)
    # model = inference(args=None, resume='/nfs/data3/koner/data/checkpoints/vit_detr/exp/skip_connection_592_432/checkpoint.pth',skip_connection=True)
    # print(model)
    # print("done")
