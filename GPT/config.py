# GPT/config.py

from pathlib import Path
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class GPTConfig(BaseSettings):

    # -----------------------
    # Model architecture
    # -----------------------
    vocab_size: int = Field(default=50257, gt=0)
    block_size: int = Field(default=1024, gt=0)

    n_layer: int = Field(default=12, gt=0)
    n_head: int = Field(default=12, gt=0)
    n_embedd: int = Field(default=768, gt=0)

    dropout: float = Field(default=0.1, ge=0.0, le=1.0)

    d_ff: int = Field(default=3072, gt=0)

    # -----------------------
    # Optimizer
    # -----------------------
    learning_rate: float = Field(default=3e-4, gt=0)
    grad_clip_norm: float = Field(default=1.0, gt=0)

    # -----------------------
    # Scheduler
    # -----------------------
    warmup_steps: int = Field(default=2000, ge=0)
    total_steps: int = Field(default=100000, gt=0)

    # -----------------------
    # Training
    # -----------------------
    batch_size: int = Field(default=8, gt=0)
    epochs: int = Field(default=1, gt=0)
    accum_steps: int = Field(default=4, gt=0)

    # -----------------------
    # Directories
    # -----------------------
    train_dir: str = "GPT/data/tfrecords/train"
    val_dir: str = "GPT/data/tfrecords/val"


    weight_decay: float = Field(default=0.1, ge=0)

    eval_interval: int = Field(default=100, gt=0)

    save_interval: int = Field(default=1000, gt=0)

    log_interval: int = Field(default=10, gt=0)

    seed: int = 42

    device: str = "gpu"

    mixed_precision: bool = True

    checkpoint_dir: str = "checkpoints"

    log_dir: str = "logs"

    num_workers: int = 4
    @field_validator("n_head")
    @classmethod
    def validate_heads(cls, v, info):
        n_embedd = info.data.get("n_embedd")

        if n_embedd is not None and n_embedd % v != 0:
            raise ValueError(
                f"n_embedd ({n_embedd}) must be divisible by n_head ({v})"
            )
        return v

    @field_validator("warmup_steps")
    @classmethod
    def validate_warmup(cls, v, info):
        total_steps = info.data.get("total_steps")

        if total_steps is not None and v >= total_steps:
            raise ValueError(
                "warmup_steps must be smaller than total_steps"
            )
        return v

    @field_validator("train_dir", "val_dir")
    @classmethod
    def validate_paths(cls, v):
        Path(v).mkdir(parents=True, exist_ok=True)
        return str(v)

    class Config:
        env_prefix = "GPT_"