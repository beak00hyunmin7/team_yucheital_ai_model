"""
추론 스크립트: 등고선 이미지 한 장을 넣으면 예측된 배수 이미지를 저장한다.

사용법 (ai_model/ 디렉터리에서):
    python -m src.inference --checkpoint checkpoints/epoch_0200.pt \
        --input path/to/contour.png --output outputs/predicted.png \
        --mode unet --image_size 256
"""

import argparse
import os

import torch
from PIL import Image
from torchvision import transforms

from src.models.translation_model import TranslationModel
from src.utils.image_utils import tensor_to_uint8


def load_input_tensor(path, image_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0)  # (1, 3, H, W)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TranslationModel(mode=args.mode, in_channels=3, out_channels=3,
                              image_size=args.image_size, device=device)
    model.load_checkpoint(args.checkpoint, map_location=device)
    model.netG.eval()

    input_t = load_input_tensor(args.input, args.image_size).to(device)
    with torch.no_grad():
        pred_t = model.netG(input_t)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    Image.fromarray(tensor_to_uint8(pred_t[0])).save(args.output)
    print(f"저장 완료: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="outputs/predicted.png")
    parser.add_argument("--mode", type=str, default="unet", choices=["unet", "gan"])
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()

    main(args)
