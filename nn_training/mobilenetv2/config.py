from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "nn_training" / "datasets"
DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
DEFAULT_RUNS_DIR = Path(__file__).resolve().parent / "runs"


@dataclass
class TrainConfig:
    data_dir: Path = DEFAULT_DATA_DIR
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR
    runs_dir: Path = DEFAULT_RUNS_DIR
    run_name: str = "symbols_mobilenetv2"
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 25
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    dropout: float = 0.20
    num_workers: int = 0
    seed: int = 42
    patience: int = 8
    device: str = "auto"
    use_class_weights: bool = True
    pretrained: bool = False
    freeze_features: bool = False


CLASS_NAMES = ("circle", "square", "star", "triangle")
CLASS_FOLDER_NAMES = ("circles", "squares", "stars", "triangles")
TRAIN_SPLIT = "trainset"
TEST_SPLIT = "testset"
VALID_SPLIT_CANDIDATES = ("validset", "validateset")
