from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .config import CLASS_FOLDER_NAMES, CLASS_NAMES, TEST_SPLIT, TRAIN_SPLIT, VALID_SPLIT_CANDIDATES


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ConvertToRGB:
    def __call__(self, image):
        return image.convert("RGB")


def resolve_split_dir(data_dir: Path, split: str) -> Path:
    data_dir = Path(data_dir).expanduser().resolve()
    if split in VALID_SPLIT_CANDIDATES:
        for candidate in VALID_SPLIT_CANDIDATES:
            path = data_dir / candidate
            if path.is_dir():
                return path
        raise FileNotFoundError(
            "Missing validation split. Expected one of: "
            + ", ".join(str(data_dir / name) for name in VALID_SPLIT_CANDIDATES)
        )

    path = data_dir / split
    if not path.is_dir():
        raise FileNotFoundError(f"Missing dataset split directory: {path}")
    return path


def count_images_by_class(
    split_dir: Path,
    class_names: Iterable[str] = CLASS_NAMES,
    folder_names: Iterable[str] = CLASS_FOLDER_NAMES,
) -> Dict[str, int]:
    counts = {}
    for class_name, folder_name in zip(class_names, folder_names):
        class_dir = split_dir / folder_name
        if not class_dir.is_dir():
            counts[class_name] = 0
            continue
        counts[class_name] = sum(
            1 for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
    return counts


def validate_dataset_layout(data_dir: Path) -> None:
    split_names = (TRAIN_SPLIT, TEST_SPLIT, VALID_SPLIT_CANDIDATES[0])
    for split_name in split_names:
        split_dir = resolve_split_dir(data_dir, split_name)
        missing_classes = [
            class_name for class_name, folder_name in zip(CLASS_NAMES, CLASS_FOLDER_NAMES)
            if not (split_dir / folder_name).is_dir()
        ]
        if missing_classes:
            raise FileNotFoundError(
                f"{split_dir} is missing class folders: {', '.join(missing_classes)}"
            )

        empty_classes = [
            class_name for class_name, count in count_images_by_class(split_dir).items()
            if count == 0
        ]
        if empty_classes:
            raise ValueError(
                f"{split_dir} has no images for classes: {', '.join(empty_classes)}"
            )


def build_transforms(image_size: int, training: bool):
    base = [
        ConvertToRGB(),
        transforms.Resize((image_size, image_size)),
    ]
    if training:
        base.extend(
            [
                transforms.RandomRotation(degrees=8),
                transforms.RandomAffine(degrees=0, translate=(0.04, 0.04), scale=(0.95, 1.05)),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
            ]
        )
    base.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transforms.Compose(base)


def _load_image_folder(split_dir: Path, image_size: int, training: bool) -> datasets.ImageFolder:
    dataset = datasets.ImageFolder(
        root=str(split_dir),
        transform=build_transforms(image_size=image_size, training=training),
    )
    expected = list(CLASS_NAMES)
    expected_folders = list(CLASS_FOLDER_NAMES)
    if dataset.classes != expected_folders:
        raise ValueError(
            f"Unexpected class folders in {split_dir}. "
            f"Expected {expected_folders}, got {dataset.classes}."
        )
    dataset.folder_classes = dataset.classes
    dataset.classes = expected
    dataset.class_to_idx = {class_name: index for index, class_name in enumerate(expected)}
    return dataset


def build_datasets(data_dir: Path, image_size: int) -> Tuple[datasets.ImageFolder, datasets.ImageFolder, datasets.ImageFolder]:
    validate_dataset_layout(data_dir)
    train_dir = resolve_split_dir(data_dir, TRAIN_SPLIT)
    valid_dir = resolve_split_dir(data_dir, VALID_SPLIT_CANDIDATES[0])
    test_dir = resolve_split_dir(data_dir, TEST_SPLIT)

    train_dataset = _load_image_folder(train_dir, image_size=image_size, training=True)
    valid_dataset = _load_image_folder(valid_dir, image_size=image_size, training=False)
    test_dataset = _load_image_folder(test_dir, image_size=image_size, training=False)
    return train_dataset, valid_dataset, test_dataset


def build_single_split_dataset(data_dir: Path, split: str, image_size: int) -> datasets.ImageFolder:
    split_dir = resolve_split_dir(data_dir, split)
    return _load_image_folder(split_dir, image_size=image_size, training=False)


def build_dataloaders(
    data_dir: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    device: Optional[torch.device] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset, valid_dataset, test_dataset = build_datasets(data_dir, image_size)
    pin_memory = bool(device is not None and device.type == "cuda")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, valid_loader, test_loader


def build_split_loader(
    data_dir: Path,
    split: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    device: Optional[torch.device] = None,
) -> DataLoader:
    dataset = build_single_split_dataset(data_dir, split, image_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=bool(device is not None and device.type == "cuda"),
    )
