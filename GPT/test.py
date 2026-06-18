#GPT/test.py

import tensorflow as tf
from config import GPTConfig
from layers.embedding import GPTEmbeddings
from layers.attention import MultiHeadAttention
from model.gpt import Decoder
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

input_ids = tf.random.uniform((2, 16), maxval=50257, dtype=tf.int32)

model = Decoder(config)

logits = model(input_ids)

print(logits.shape)