# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
import random
from pathlib import Path
import numpy as np
import torch
import torch.utils.data
import torchvision
from torchvision import transforms
from pycocotools import mask as coco_mask
import albumentations as A
import datasets.transforms as T
import torchvision.transforms.functional as F
from datasets.SmallObjectAugmentation import SmallObjectAugmentation

SOA_THRESH = 80 * 80
SOA_PROB = 0.5
SOA_COPY_TIMES = 3
SOA_EPOCHS = 30
SOA_ONE_OBJECT = False
SOA_ALL_OBJECTS = False

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, small_augment, color_augmentation=None, spatial_augmentation=None):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.small_augment = small_augment
        self.spatial_augmentation = spatial_augmentation
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

        # Convert PIL image to numpy array, the axis are different so need to transpose
        img = np.array(img).transpose(1, 0, 2)

        # Copy small objects multiple times randomly
        if self.small_augment:
            sample = self._augmentation(img, target)
            if sample is not None:
                img, target = sample['img'], sample['target']

        # Spacial transformations/ augmentations
        if self.spatial_augmentation is not None:
            # Albumentation expects height first and then width in numpy array, so need to transpose.
            img = img.transpose(1, 0, 2)
            transformed = self.spatial_augmentation(image=img, bboxes=target['boxes'], category_ids=target['labels'])
            img = transformed['image']
            target['boxes'] = torch.tensor(transformed['bboxes'], dtype=torch.float32)
            # Albumentation returns height first tensor, we need to make it width first
            img = img.permute(0, 2, 1)


        # if self._transforms is not None:
        #     # This transformation expects images to be in PIL format. So need too transpose because numpy and PIL axis are different
        #     img = img.transpose(1, 0, 2)
        #     img = transforms.ToPILImage()(img)
        #     img, target = self._transforms(img, target)
        #
        # # Color/ pixes wise augmentations
        # if self.color_augmentation is not None:
        #     img = self.color_augmentation(image=img)["image"]
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
        color_aug_list = [
            A.RandomBrightnessContrast(), A.RandomBrightnessContrast(contrast_limit=0.),
            A.RandomBrightnessContrast(brightness_limit=0.), A.RGBShift(), A.HueSaturationValue(),
            A.ChannelShuffle(), A.CLAHE(), A.RandomGamma(), A.Blur(), A.ToGray(), A.ToSepia(),]
        # Return random augmentation with 0.7 probability
        return random.choices([A.NoOp(), random.choice(color_aug_list)],  weights=[0.3, 0.7])[0]

    elif image_set == 'val':
        return A.NoOp()

    raise ValueError(f'unknown {image_set}')

def spatial_augmentation(image_set, image_size):
    def toTensor(img, **params):
        return F.to_tensor(img)

    normalize = A.Compose([A.Resize(image_size[1], image_size[0]), #height first for this lobrary
                          A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                          A.Lambda(p=1, image=toTensor)])

    if image_set == 'train': #todo this 384 is final or we sud think more on size
        spacial_aug_list = [A.HorizontalFlip(), A.Flip(), #vertical
            A.Transpose(), A.RandomRotate90(), A.RandomSizedBBoxSafeCrop(384, 600),
            A.ShiftScaleRotate(), A.LongestMaxSize(),]
        random_color_aug = color_augmentation(image_set)
        random_spacial_aug = random.choices([A.NoOp(), random.choice(spacial_aug_list)], weights=[0.3, 0.7])[0]
        transform = A.Compose([random_spacial_aug, random_color_aug, normalize],
                    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))
        return transform

    if image_set == 'val':
        return A.Compose(normalize,
                bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']),)

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
            A.Transpose(),
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
            dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks, small_augment=args.small_augment, color_augmentation = color_augment)
        else:
            dataset = CocoDetection(img_folder, ann_file, transforms=None,#make_coco_transforms_ViT(image_set, args.img_size, None),
                                    return_masks=args.masks, small_augment=args.small_augment,
                                    color_augmentation = color_augment, spatial_augmentation=spatial_augmentation(image_set, args.img_size))
    else:
        dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks, aug=args.augment)

    return dataset
