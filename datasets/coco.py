# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
import random
from pathlib import Path
import torch
import torch.utils.data
import torchvision
from torchvision import transforms
from pycocotools import mask as coco_mask
import albumentations as A
import datasets.transforms as T
from datasets.SmallObjectAugmentation import SmallObjectAugmentation

SOA_THRESH = 80 * 80
SOA_PROB = 0.5
SOA_COPY_TIMES = 3
SOA_EPOCHS = 30
SOA_ONE_OBJECT = False
SOA_ALL_OBJECTS = False

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, small_augment, color_augmentation=None,image_set='train'):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.image_set = image_set
        self.small_augment = small_augment
        self._transforms = transforms
        self.color_augmentation = color_augmentation
        if small_augment:
            self._augmentation = SmallObjectAugmentation(SOA_THRESH, SOA_PROB, SOA_COPY_TIMES,
                                                         SOA_EPOCHS, SOA_ALL_OBJECTS, SOA_ONE_OBJECT)
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)

        # Copy small objects multiple times randomly
        if self.small_augment and self.image_set=='train':
            sample = self._augmentation(img, target)
            if sample is not None:
                img, target = sample['img'], sample['target']
                img = transforms.ToPILImage()(img)

        # Spacial transformations/ augmentations
        if self._transforms is not None:
            img, target = self._transforms(img, target)

        # Color/ pixes wise augmentations
        if self.color_augmentation is not None:
            img = self.color_augmentation(image=img)["image"]
        return img, target


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

    def __call__(self, image, target):
        w, h = image.size

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

        return image, target


def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')

def color_augmentation(image_set):
    if image_set == 'train':
        aug_list = [
            A.RandomBrightnessContrast(), A.RandomBrightnessContrast(contrast_limit=0.),
            A.RandomBrightnessContrast(brightness_limit=0.), A.RGBShift(), A.HueSaturationValue(),
            A.ChannelShuffle(), A.CLAHE(), A.RandomGamma(), A.Blur(), A.ToGray(), A.ToSepia(),]
        # Return random augmentation with 0.5 probability
        return random.choice([A.NoOp(), random.choice(aug_list)])

    elif image_set == 'val':
        return A.NoOp()

    raise ValueError(f'unknown {image_set}')

def make_coco_transforms_ViT(image_set, image_size, max=None):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train': #todo this 384 is final or we sud think more on size
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                # T.RandomResize(scales, max_size=1333),
                T.FixedResize(image_size, max),
                T.Compose([
                    T.RandomResize(scales),
                    T.RandomSizeCrop(384, 600),
                    # T.RandomResize(scales, max_size=1333),
                    T.FixedResize(image_size, max),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            # T.RandomResize([800], max_size=1333),
            T.FixedResize(image_size, max),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


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
    color_augment = color_augmentation(image_set) if args.color_augment else None
    if args.backbone in ("ViT", "Deit"):
        if args.random_image_size:
            dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks, small_augment=args.small_augment, color_augmentation = color_augment,image_set=image_set)
        else:
            dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms_ViT(image_set, args.img_size, None), return_masks=args.masks, small_augment=args.small_augment, color_augmentation = color_augment,image_set=image_set)
    else:
        dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks, aug=args.augment, image_set=image_set)

    return dataset
