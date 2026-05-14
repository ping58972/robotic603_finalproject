import csv
import json
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device_name: str) -> torch.device:
    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def append_history_row(path: Path, row: Dict[str, Any], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def class_weights_from_targets(targets: Iterable[int], num_classes: int) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for target in targets:
        counts[int(target)] += 1
    if torch.any(counts == 0):
        raise ValueError(f"Cannot compute class weights with empty classes: {counts.tolist()}")
    return counts.sum() / (num_classes * counts)


def confusion_matrix(predictions: Iterable[int], targets: Iterable[int], num_classes: int) -> torch.Tensor:
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for target, predicted in zip(targets, predictions):
        matrix[int(target), int(predicted)] += 1
    return matrix


def classification_report(matrix: torch.Tensor, class_names: Sequence[str]) -> List[Dict[str, Any]]:
    rows = []
    total_correct = 0
    total_count = int(matrix.sum().item())
    for index, class_name in enumerate(class_names):
        true_positive = int(matrix[index, index].item())
        false_positive = int(matrix[:, index].sum().item() - true_positive)
        false_negative = int(matrix[index, :].sum().item() - true_positive)
        support = int(matrix[index, :].sum().item())
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        total_correct += true_positive
        rows.append(
            {
                "class": class_name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )
    accuracy = total_correct / total_count if total_count else 0.0
    rows.append(
        {
            "class": "overall",
            "precision": accuracy,
            "recall": accuracy,
            "f1": accuracy,
            "support": total_count,
        }
    )
    return rows


def print_report(rows: Sequence[Dict[str, Any]]) -> None:
    print(f"{'class':<12} {'precision':>10} {'recall':>10} {'f1':>10} {'support':>10}")
    for row in rows:
        print(
            f"{row['class']:<12} "
            f"{row['precision']:>10.4f} "
            f"{row['recall']:>10.4f} "
            f"{row['f1']:>10.4f} "
            f"{row['support']:>10}"
        )

