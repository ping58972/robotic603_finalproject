import argparse
from pathlib import Path
from typing import List, Tuple

import torch
from torch import nn

from .config import CLASS_NAMES, DEFAULT_CHECKPOINT_DIR, DEFAULT_DATA_DIR, DEFAULT_RUNS_DIR
from .data import build_split_loader
from .model import build_model
from .utils import classification_report, confusion_matrix, print_report, save_json, select_device


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained symbol CNN checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_DIR / "symbols_cnn_best.pt")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split", default="testset", choices=("trainset", "validset", "validateset", "testset"))
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=DEFAULT_RUNS_DIR / "evaluation_report.json")
    return parser.parse_args()


def evaluate_model(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, List[int], List[int]]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            predictions = outputs.argmax(dim=1)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (predictions == labels).sum().item()
            total_count += batch_size
            all_predictions.extend(predictions.cpu().tolist())
            all_targets.extend(labels.cpu().tolist())

    return total_loss / total_count, total_correct / total_count, all_predictions, all_targets


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    class_names = checkpoint.get("class_names", list(CLASS_NAMES))

    model = build_model(num_classes=len(class_names), dropout=0.0).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    loader = build_split_loader(
        data_dir=args.data_dir,
        split=args.split,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    criterion = nn.CrossEntropyLoss()
    loss, accuracy, predictions, targets = evaluate_model(model, loader, criterion, device)
    matrix = confusion_matrix(predictions, targets, num_classes=len(class_names))
    report = classification_report(matrix, class_names)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split: {args.split}")
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print_report(report)

    payload = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "loss": loss,
        "accuracy": accuracy,
        "class_names": class_names,
        "confusion_matrix": matrix.tolist(),
        "report": report,
    }
    save_json(args.output, payload)
    print(f"Saved report: {args.output}")


if __name__ == "__main__":
    main()
