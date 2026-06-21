#GPT/tfrecords.py

import tensorflow as tf
import os

from GPT.config import GPTConfig
from GPT.data.dataset_download import download_ds
from GPT.data.sample_builder import build_samples
from GPT.data.tfrecord_writer import shard_tfrecords

if __name__ == "__main__":
    config=GPTConfig()

    os.makedirs("GPT/data", exist_ok=True)
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

    path = "GPT/data/shakespeare.txt"

    download_ds(url,path)

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    samples = build_samples(text, config.block_size)


    shard_tfrecords(
        samples,
        config,
        num_shards=10,
        validation_split=0.2
    )

    