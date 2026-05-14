import argparse
from pathlib import Path
from typing import Iterable, List

import torch
from PIL import Image

from .config import CLASS_NAMES, DEFAULT_CHECKPOINT_DIR
from .data import IMAGE_SUFFIXES, build_transforms
from .model import build_model
from .utils import select_device


def parse_args():
    parser = argparse.ArgumentParser(description="Predict symbol classes for image files.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_DIR / "symbols_cnn_best.pt")
    parser.add_argument("--image", type=Path, help="Path to one image.")
    parser.add_argument("--image-dir", type=Path, help="Path to a directory of images.")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def iter_images(image: Path, image_dir: Path) -> Iterable[Path]:
    if image is not None:
        yield image
    if image_dir is not None:
        for path in sorted(image_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                yield path


def load_checkpoint_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint.get("class_names", list(CLASS_NAMES))
    config = checkpoint.get("config", {})
    dropout = float(config.get("dropout", 0.0)) if isinstance(config, dict) else 0.0
    model = build_model(num_classes=len(class_names), dropout=dropout).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, class_names


def predict_image(model, class_names: List[str], image_path: Path, transform, top_k: int, device: torch.device):
    with Image.open(image_path) as image:
        tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1).squeeze(0)
    limit = min(top_k, len(class_names))
    scores, indexes = torch.topk(probabilities, k=limit)
    return [(class_names[int(index)], float(score)) for score, index in zip(scores.cpu(), indexes.cpu())]


def main() -> None:
    args = parse_args()
    if args.image is None and args.image_dir is None:
        raise SystemExit("Provide --image or --image-dir.")

    device = select_device(args.device)
    model, class_names = load_checkpoint_model(args.checkpoint, device)
    transform = build_transforms(args.image_size, training=False)

    for image_path in iter_images(args.image, args.image_dir):
        predictions = predict_image(model, class_names, image_path, transform, args.top_k, device)
        formatted = ", ".join(f"{label}={score:.4f}" for label, score in predictions)
        print(f"{image_path}: {formatted}")


if __name__ == "__main__":
    main()
