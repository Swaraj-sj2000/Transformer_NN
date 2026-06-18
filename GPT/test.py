#!/bin/python
#GPT/test.py

import tensorflow as tf
from config import GPTConfig
from embedding import GPTEmbeddings
from causal_Attention import MultiHeadAttention
config = GPTConfig()

emb = GPTEmbeddings(config)

x = tf.constant([
    [10, 20, 30, 40],
    [50, 60, 70, 80]
])

y = emb(x)

print(y.shape)

x = tf.random.normal((2,16,768))

attn = MultiHeadAttention(config)

y = attn(x)

print(y.shape)