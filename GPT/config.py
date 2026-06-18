#!/bin/python
#GPT/config.py
# config.py


from pydantic_settings import BaseSettings


class GPTConfig(BaseSettings):
    vocab_size: int = 50257
    block_size: int = 1024

    n_layer: int = 12
    n_head: int = 12
    n_embedd: int = 768

    dropout: float = 0.1

    class Config:
        env_prefix = "GPT_"