import json
from PIL import Image
import torch
from torchvision import transforms
from models.pytorch_pretrained_vit.model import ViT as pre_trained_ViT
from models.pytorch_pretrained_vit.configs import PRETRAINED_MODELS


def clsssify():
    # Load ViT
    # model = ViT('B_16_imagenet1k', pretrained=True)
    model_vit = pre_trained_ViT('B_16_imagenet1k',
                                pretrained=True,
                                weight_path="/mnt/data/hannan/.cache/torch/checkpoints/B_16_imagenet1k.pth",
                                detr_compatibility=True,
                                image_size=384,
                                )
    model_vit.eval()

    # Load image
    # NOTE: Assumes an image `img.jpg` exists in the current directory
    img = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])(Image.open('/mnt/data/hannan/detr-out/img.jpg')).unsqueeze(0)
    print(img.shape)  # torch.Size([1, 3, 384, 384])

    # Classify
    with torch.no_grad():
        out, pos = model_vit(img)
    print(out.shape)  # ([1, 577, 768]) dim=786
    print(pos.shape)  # ([1, 577, 768])


def main():
    # Downloading: "https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_16_imagenet1k.pth" to /mnt/data/hannan/.cache/torch/checkpoints/B_16_imagenet1k.pth
    # model = ViT('B_16_imagenet1k', pretrained=True)
    # model_vit = ViT('B_16_imagenet1k', pretrained=True,
    #                 weight_path="/mnt/data/hannan/.cache/torch/checkpoints/B_16_imagenet1k.pth")
    # print(model_vit)
    clsssify()
    # print(PRETRAINED_MODELS['B_16_imagenet1k']['config']['dim'])

    # backbone = "resnet"
    #
    # print(backbone != "ViT" and backbone not in PRETRAINED_MODELS.keys())


if __name__ == '__main__':
    main()
