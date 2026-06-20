#GPT/training/trainer.py

import tensorflow as tf
from config import GPTConfig

config=GPTConfig()
clip_norm=config.grad_clip_norm
accum_steps=config.accum_steps

@tf.function
def train_step(model, optimizer, loss_fn,x):
    with tf.GradientTape() as tape:
        print("before model")
        logits = model(x)
        print("after model")
        targets = x[:, 1:]
        logits = logits[:, :-1, :]

        loss = loss_fn(logits, targets)
        print("after loss")
    print("before gradients")
    grads = tape.gradient(loss, model.trainable_variables)
    print("After gradients")
    grads,_=tf.clip_by_global_norm(grads,clip_norm)
    print("Before optimizer")
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    print("after optimizer")

    return loss