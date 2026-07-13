#GPT/training/evaluation_loop.py

import tensorflow as tf
import numpy as np

def evaluate_perplexity(model,dataset,loss_fn):
    total_loss=0.0
    total_tokens=0.0

    for x in dataset:
        logits=model(x)

        targets=x[:,1:]
        logits=logits[:,:-1,:]
        loss=loss_fn(targets,logits)
        batch_tokens=tf.size(targets)

        total_loss+=tf.reduce_sum(loss).numpy()
        total_tokens+=batch_tokens.numpy()

    avg_loss=total_loss/total_tokens
    ppl=np.exp(avg_loss)

    return ppl