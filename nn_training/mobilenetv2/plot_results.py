import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import CLASS_NAMES, DEFAULT_RUNS_DIR


def parse_args():
    default_run_dir = DEFAULT_RUNS_DIR / "symbols_mobilenetv2"
    parser = argparse.ArgumentParser(description="Plot MobileNetV2 train, validation, and test results.")
    parser.add_argument("--run-dir", type=Path, default=default_run_dir)
    parser.add_argument("--history", type=Path, help="Path to history.csv.")
    parser.add_argument("--dataset-counts", type=Path, help="Path to dataset_counts.json.")
    parser.add_argument("--test-metrics", type=Path, help="Path to test_metrics.json.")
    parser.add_argument("--evaluation-report", type=Path, default=DEFAULT_RUNS_DIR / "evaluation_report.json")
    parser.add_argument("--output", type=Path, help="Output image path.")
    return parser.parse_args()


def load_history(path: Path) -> List[Dict[str, float]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = []
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "epoch": int(row["epoch"]),
                    "train_loss": float(row["train_loss"]),
                    "train_accuracy": float(row["train_accuracy"]),
                    "valid_loss": float(row["valid_loss"]),
                    "valid_accuracy": float(row["valid_accuracy"]),
                    "learning_rate": float(row["learning_rate"]),
                }
            )
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_input_paths(args) -> Dict[str, Path]:
    run_dir = args.run_dir
    return {
        "history": args.history or run_dir / "history.csv",
        "dataset_counts": args.dataset_counts or run_dir / "dataset_counts.json",
        "test_metrics": args.test_metrics or run_dir / "test_metrics.json",
        "evaluation_report": args.evaluation_report,
        "output": args.output or run_dir / "results_summary.png",
    }


def plot_curves(ax, epochs, train_values, valid_values, title, ylabel):
    ax.plot(epochs, train_values, marker="o", linewidth=2, label="train")
    ax.plot(epochs, valid_values, marker="o", linewidth=2, label="valid")
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()


def plot_dataset_counts(ax, counts: Dict[str, Dict[str, int]], class_names: List[str]):
    splits = ["trainset", "validset", "testset"]
    x = list(range(len(class_names)))
    width = 0.25
    offsets = [-width, 0, width]

    for split, offset in zip(splits, offsets):
        values = [counts.get(split, {}).get(class_name, 0) for class_name in class_names]
        ax.bar([index + offset for index in x], values, width=width, label=split)

    ax.set_title("Dataset Images By Class")
    ax.set_xlabel("class")
    ax.set_ylabel("images")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=20, ha="right")
    ax.legend()


def plot_final_metrics(ax, history: List[Dict[str, float]], test_metrics: Dict[str, Any]):
    last = history[-1]
    labels = ["train", "valid", "test"]
    accuracies = [
        last["train_accuracy"],
        last["valid_accuracy"],
        float(test_metrics.get("test_accuracy", 0.0)),
    ]
    losses = [
        last["train_loss"],
        last["valid_loss"],
        float(test_metrics.get("test_loss", 0.0)),
    ]

    x = list(range(len(labels)))
    ax.bar([index - 0.18 for index in x], accuracies, width=0.36, label="accuracy")
    ax.bar([index + 0.18 for index in x], losses, width=0.36, label="loss")
    ax.set_title("Final Train, Valid, Test")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(max(accuracies), max(losses), 1.0) * 1.10)
    ax.legend()

    for index, value in enumerate(accuracies):
        ax.text(index - 0.18, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    for index, value in enumerate(losses):
        ax.text(index + 0.18, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)


def plot_confusion_matrix(ax, evaluation_report: Optional[Dict[str, Any]], class_names: List[str]):
    if not evaluation_report or "confusion_matrix" not in evaluation_report:
        ax.axis("off")
        ax.text(0.5, 0.5, "No evaluation report found", ha="center", va="center")
        return

    matrix = evaluation_report["confusion_matrix"]
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title("Test Confusion Matrix")
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)

    max_value = max(max(row) for row in matrix) if matrix else 0
    threshold = max_value / 2 if max_value else 0
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            color = "white" if value > threshold else "black"
            ax.text(col_index, row_index, str(value), ha="center", va="center", color=color)

    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def plot_results(paths: Dict[str, Path]) -> None:
    history = load_history(paths["history"])
    counts = load_json(paths["dataset_counts"]) or {}
    test_metrics = load_json(paths["test_metrics"]) or {}
    evaluation_report = load_json(paths["evaluation_report"])

    class_names = list(evaluation_report.get("class_names", CLASS_NAMES)) if evaluation_report else list(CLASS_NAMES)
    epochs = [row["epoch"] for row in history]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    fig.suptitle("Symbol MobileNetV2 Training Results", fontsize=18)

    plot_curves(
        axes[0][0],
        epochs,
        [row["train_accuracy"] for row in history],
        [row["valid_accuracy"] for row in history],
        "Accuracy",
        "accuracy",
    )
    plot_curves(
        axes[0][1],
        epochs,
        [row["train_loss"] for row in history],
        [row["valid_loss"] for row in history],
        "Loss",
        "loss",
    )

    axes[0][2].plot(epochs, [row["learning_rate"] for row in history], marker="o", linewidth=2)
    axes[0][2].set_title("Learning Rate")
    axes[0][2].set_xlabel("epoch")
    axes[0][2].set_ylabel("learning rate")
    axes[0][2].grid(True, alpha=0.25)

    plot_dataset_counts(axes[1][0], counts, class_names)
    plot_final_metrics(axes[1][1], history, test_metrics)
    plot_confusion_matrix(axes[1][2], evaluation_report, class_names)

    output = paths["output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)

    best_valid = max(history, key=lambda row: row["valid_accuracy"])
    print(f"Saved graph: {output}")
    print(
        "Best validation accuracy: "
        f"{best_valid['valid_accuracy']:.4f} at epoch {best_valid['epoch']}"
    )
    if test_metrics:
        print(
            "Test result: "
            f"accuracy={float(test_metrics.get('test_accuracy', 0.0)):.4f}, "
            f"loss={float(test_metrics.get('test_loss', 0.0)):.4f}"
        )


def main() -> None:
    paths = resolve_input_paths(parse_args())
    plot_results(paths)


if __name__ == "__main__":
    main()
