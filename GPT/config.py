#GPT/config.py
from pydantic_settings import BaseSettings


class GPTConfig(BaseSettings):

    # -----------------------
    # Model architecture
    # -----------------------
    vocab_size: int = 50257
    block_size: int = 1024

    n_layer: int = 12
    n_head: int = 12
    n_embedd: int = 768

    dropout: float = 0.1

    # derived MLP size (important for SwiGLU / FFN)
    d_ff: int = 3072   # 4 * 768 (GPT-2 style)

    # -----------------------
    # Optimizer
    # -----------------------
    learning_rate: float = 3e-4
    grad_clip_norm: float = 1.0

    # -----------------------
    # Scheduler
    # -----------------------
    warmup_steps: int = 2000
    total_steps: int = 100000

    # -----------------------
    # Training
    # -----------------------
    batch_size: int = 8
    epochs: int = 1
    accum_steps:int=4

    # -----------------------
    #DIRECTORY
    # -----------------------
    train_dir:str="GPT/data/tfrecords/train"
    val_dir:str="GPT/data/tfrecords/val"




    class Config:
        env_prefix = "GPT_"