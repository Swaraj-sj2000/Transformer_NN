# GPT/config.py
from pydantic import  Field, ConfigDict, field_validator
from pydantic_settings import BaseSettings

from pathlib import Path

class GPTConfig(BaseSettings):
    """Configuration for GPT-2 model training.
    
    Validates all hyperparameters with sensible defaults.
    Can load from environment variables with GPT_ prefix.
    """
    model_config = ConfigDict(env_prefix="GPT_")

    # ============ Model Architecture ============
    vocab_size: int = Field(default=50257, gt=0, description="GPT-2 vocabulary size")
    block_size: int = Field(default=128, gt=0, description="Max sequence length")
    n_layer: int = Field(default=12, gt=0, description="Number of transformer blocks")
    n_head: int = Field(default=12, gt=0, description="Number of attention heads")
    n_embedd: int = Field(default=768, gt=0, description="Embedding dimension")
    d_ff: int = Field(default=3072, gt=0, description="Feed-forward hidden dimension")
    epsilon: float = Field(default=1e-12, ge=0.0, description="LayerNorm epsilon")
    dropout: float = Field(default=0.1, ge=0.0, le=1.0, description="Dropout rate")

    # ============ Training Hyperparameters ============
    batch_size: int = Field(default=8, gt=0, description="Batch size for training")
    learning_rate: float = Field(default=3e-4, gt=0, description="Initial learning rate")
    weight_decay: float = Field(default=0.1, ge=0, description="L2 regularization")
    grad_clip_norm: float = Field(default=1.0, gt=0, description="Gradient clipping threshold")
    warmup_steps: int = Field(default=2000, ge=0, description="Learning rate warmup steps")
    total_steps: int = Field(default=100000, gt=0, description="Total training steps")
    accum_steps: int = Field(default=4, gt=0, description="Gradient accumulation steps")

    # ============ Data & Paths ============
    train_dir: str = Field(default="GPT/data/tfrecords/train", description="Training TFRecord directory")
    val_dir: str = Field(default="GPT/data/tfrecords/val", description="Validation TFRecord directory")
    checkpoint_dir: str = Field(default="checkpoints", description="Checkpoint save directory")
    log_dir: str = Field(default="logs", description="Tensorboard log directory")

    # ============ Evaluation & Logging ============
    eval_interval: int = Field(default=100, gt=0, description="Eval frequency (steps)")
    save_interval: int = Field(default=1000, gt=0, description="Checkpoint save frequency (steps)")
    log_interval: int = Field(default=10, gt=0, description="Logging frequency (steps)")
    num_workers: int = Field(default=4, gt=0, description="Data pipeline workers")
    #=========== Sampler =============
    temperature: float = Field(default=1.0, ge=0.0, description="Sampling temperature")
    top_k: int = Field(default=50, ge=0, description="Top-k sampling")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p (nucleus) sampling")

    # ============ Validation ============
    @field_validator("block_size")
    @classmethod
    def validate_block_size(cls, v):
        if v & (v - 1):  # Check if power of 2
            raise ValueError(f"block_size {v} should be a power of 2")
        return v

    @field_validator("n_head")
    @classmethod
    def validate_heads(cls, v, info):
        n_embedd = info.data.get("n_embedd")
        if n_embedd is not None and n_embedd % v != 0:
            raise ValueError(f"n_embedd ({n_embedd}) must be divisible by n_head ({v})")
        return v

    @field_validator("warmup_steps")
    @classmethod
    def validate_warmup(cls, v, info):
        total_steps = info.data.get("total_steps")
        if total_steps is not None and v >= total_steps:
            raise ValueError("warmup_steps must be smaller than total_steps")
        return v

    @field_validator("train_dir", "val_dir", "checkpoint_dir", "log_dir")
    @classmethod
    def validate_paths(cls, v):
        """Create directories if they don't exist."""
        Path(v).mkdir(parents=True, exist_ok=True)
        return str(v)

if __name__ == "__main__":
    # Initialize global config
    config = GPTConfig()
    print(f"  Config loaded")
    print(f"  Model: {config.n_layer}L, {config.n_head}H, {config.n_embedd}D")
    print(f"  Batch: {config.batch_size}, LR: {config.learning_rate}, Warmup: {config.warmup_steps}")
