"""
Training Configuration for FuseForm on MCubeS Dataset
Based on: McMillen & Yilmaz, "FuseForm: Multimodal Transformer for Semantic Segmentation", WACVW 2025
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataConfig:
    """Dataset configuration"""
    root: str = "../multimodal_dataset"
    crop_size: int = 512
    num_classes: int = 20
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"
    num_workers: int = 4


@dataclass
class TrainConfig:
    """Training hyperparameters (from FuseForm paper Section 4.2)"""
    # Epochs
    num_epochs: int = 500
    warmup_epochs: int = 10

    # Batch size
    batch_size: int = 2

    # Learning rate schedule
    lr_init: float = 6e-7  # Initial learning rate
    lr_max: float = 6e-6   # Maximum learning rate after warmup
    lr_min: float = 1e-9   # Minimum learning rate

    # Optimizer (AdamW)
    optimizer: str = "adamw"
    weight_decay: float = 1e-2
    adam_epsilon: float = 1e-8

    # Scheduler (Poly)
    scheduler: str = "poly"
    poly_power: float = 0.9

    # Loss
    loss_fn: str = "cross_entropy"

    # Device
    device: str = "cuda"

    # Checkpointing
    save_interval: int = 50  # Save checkpoint every N epochs
    save_best: bool = True
    checkpoint_dir: str = "./checkpoints"


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # Encoder backbone
    encoder_type: str = "MiT"  # Mix-Transformer
    encoder_variant: str = "mit_b4"  # Options: mit_b0, mit_b1, mit_b2, mit_b3, mit_b4
    pretrained: bool = True
    pretrained_weights: str = "imagenet"

    # Modalities
    num_modalities: int = 4  # RGB, AoLP, DoLP, NIR

    # Decoder
    decoder_type: str = "FuseForm"
    decoder_blocks: int = 2  # D_i = D_j = 2 for all stages

    # Fusion module
    fusion_type: str = "hybrid"  # Global + Local fusion


@dataclass
class AugmentConfig:
    """Data augmentation configuration (from FuseForm paper)"""
    random_resized_crop: bool = True
    crop_scale: tuple = (0.5, 1.0)
    crop_ratio: tuple = (3.0/4.0, 4.0/3.0)

    color_jitter: bool = True
    brightness: float = 0.4
    contrast: float = 0.4
    saturation: float = 0.4
    hue: float = 0.1

    random_flip: bool = True
    flip_prob: float = 0.5

    gaussian_blur: bool = True
    blur_prob: float = 0.5
    blur_sigma: tuple = (0.1, 2.0)

    # Note: Random crop and horizontal flip are in dataset.py


@dataclass
class EvalConfig:
    """Evaluation configuration"""
    val_interval: int = 1  # Validate every N epochs
    metrics: list = None  # Computed metrics

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ["mIoU", "mAccuracy", "allAccuracy"]


@dataclass
class Config:
    """Master configuration"""
    data: DataConfig = None
    train: TrainConfig = None
    model: ModelConfig = None
    augment: AugmentConfig = None
    eval: EvalConfig = None
    seed: int = 42

    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.train is None:
            self.train = TrainConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.augment is None:
            self.augment = AugmentConfig()
        if self.eval is None:
            self.eval = EvalConfig()


# Default configuration
config = Config()
