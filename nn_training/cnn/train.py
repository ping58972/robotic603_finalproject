import argparse
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn

from .config import CLASS_NAMES, TrainConfig
from .data import build_dataloaders, count_images_by_class, resolve_split_dir
from .model import build_model
from .utils import append_history_row, class_weights_from_targets, save_json, select_device, set_seed, to_jsonable


def parse_args() -> TrainConfig:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="Train a CNN for symbol image classification.")
    parser.add_argument("--data-dir", type=Path, default=defaults.data_dir)
    parser.add_argument("--checkpoint-dir", type=Path, default=defaults.checkpoint_dir)
    parser.add_argument("--runs-dir", type=Path, default=defaults.runs_dir)
    parser.add_argument("--run-name", default=defaults.run_name)
    parser.add_argument("--image-size", type=int, default=defaults.image_size)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--dropout", type=float, default=defaults.dropout)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--patience", type=int, default=defaults.patience)
    parser.add_argument("--device", default=defaults.device, help="auto, cpu, cuda, cuda:0, or mps")
    parser.add_argument("--no-class-weights", action="store_true")
    args = parser.parse_args()

    return TrainConfig(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        runs_dir=args.runs_dir,
        run_name=args.run_name,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        num_workers=args.num_workers,
        seed=args.seed,
        patience=args.patience,
        device=args.device,
        use_class_weights=not args.no_class_weights,
    )


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer = None,
) -> Tuple[float, float]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        if training:
            loss.backward()
            optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_count += batch_size

    return total_loss / total_count, total_correct / total_count


def make_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    config: TrainConfig,
) -> Dict:
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": to_jsonable(config),
        "class_names": list(CLASS_NAMES),
        "class_to_idx": {class_name: index for index, class_name in enumerate(CLASS_NAMES)},
    }


def main() -> None:
    config = parse_args()
    set_seed(config.seed)
    device = select_device(config.device)

    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_dir = config.runs_dir / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_loader, valid_loader, test_loader = build_dataloaders(
        data_dir=config.data_dir,
        image_size=config.image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        device=device,
    )

    train_counts = count_images_by_class(resolve_split_dir(config.data_dir, "trainset"))
    valid_counts = count_images_by_class(resolve_split_dir(config.data_dir, "validset"))
    test_counts = count_images_by_class(resolve_split_dir(config.data_dir, "testset"))
    save_json(run_dir / "dataset_counts.json", {"trainset": train_counts, "validset": valid_counts, "testset": test_counts})
    save_json(run_dir / "config.json", config)

    model = build_model(num_classes=len(CLASS_NAMES), dropout=config.dropout).to(device)

    if config.use_class_weights:
        weights = class_weights_from_targets(train_loader.dataset.targets, num_classes=len(CLASS_NAMES)).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    print(f"Device: {device}")
    print(f"Train counts: {train_counts}")
    print(f"Valid counts: {valid_counts}")
    print(f"Test counts:  {test_counts}")

    history_path = run_dir / "history.csv"
    history_fields = (
        "epoch",
        "train_loss",
        "train_accuracy",
        "valid_loss",
        "valid_accuracy",
        "learning_rate",
    )
    best_valid_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, config.epochs + 1):
        train_loss, train_accuracy = run_epoch(model, train_loader, criterion, device, optimizer)
        with torch.no_grad():
            valid_loss, valid_accuracy = run_epoch(model, valid_loader, criterion, device)

        scheduler.step(valid_loss)
        learning_rate = optimizer.param_groups[0]["lr"]
        metrics = {
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "valid_loss": valid_loss,
            "valid_accuracy": valid_accuracy,
            "learning_rate": learning_rate,
        }
        append_history_row(history_path, {"epoch": epoch, **metrics}, history_fields)

        checkpoint = make_checkpoint(model, optimizer, epoch, metrics, config)
        latest_path = config.checkpoint_dir / f"{config.run_name}_latest.pt"
        torch.save(checkpoint, latest_path)

        improved = valid_loss < best_valid_loss
        if improved:
            best_valid_loss = valid_loss
            epochs_without_improvement = 0
            best_path = config.checkpoint_dir / f"{config.run_name}_best.pt"
            torch.save(checkpoint, best_path)
        else:
            epochs_without_improvement += 1

        print(
            f"epoch {epoch:03d}/{config.epochs:03d} "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            f"valid_loss={valid_loss:.4f} valid_acc={valid_accuracy:.4f} "
            f"lr={learning_rate:.6g}"
        )

        if epochs_without_improvement >= config.patience:
            print(f"Early stopping after {config.patience} epochs without validation loss improvement.")
            break

    best_checkpoint_path = config.checkpoint_dir / f"{config.run_name}_best.pt"
    if best_checkpoint_path.exists():
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    with torch.no_grad():
        test_loss, test_accuracy = run_epoch(model, test_loader, criterion, device)
    test_metrics = {"test_loss": test_loss, "test_accuracy": test_accuracy}
    save_json(run_dir / "test_metrics.json", test_metrics)
    print(f"Test loss={test_loss:.4f} test_acc={test_accuracy:.4f}")
    print(f"Best checkpoint: {best_checkpoint_path}")
    print(f"Training history: {history_path}")


if __name__ == "__main__":
    main()
