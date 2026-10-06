import os
import tensorflow as tf

from GPT.config import GPTConfig
from GPT.model.gpt import Decoder
from GPT.inference.sampler import Sampler


def load_model_weights(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model.load_weights(checkpoint_path)
    print(f"Loaded weights from {checkpoint_path}")
    return model


if __name__ == "__main__":
    config = GPTConfig()
    model = Decoder(config=config)
    prompt = tf.constant([[1, 2, 3]], dtype=tf.int32)

    # Build subclassed model variables before loading Keras weights.
    _ = model(prompt, training=False)

    ckpt = os.path.join(config.checkpoint_dir, "model.weights.h5")
    if os.path.exists(ckpt):
        model = load_model_weights(model, ckpt)
    else:
        print(f"No checkpoint found at {ckpt}; using randomly initialized weights.")

    sampler = Sampler(config=config)
    out = sampler.generate(
        model,
        prompt,
        max_new_tokens=8,
        decode_strategy="greedy",
        temperature=config.temperature,
        top_k=config.top_k,
        top_p=config.top_p,
    )
    print("Generated ids:", out.numpy().tolist())
