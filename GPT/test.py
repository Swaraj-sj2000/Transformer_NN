#GPT/test.py

import tensorflow as tf
from config import GPTConfig
from layers.embedding import GPTEmbeddings
from layers.attention import MultiHeadAttention
from model.gpt import Decoder
from training.loss import GPTLoss
from training.trainer import train_step
from training.optimizer import build_optimizer
import tensorflow as tf

tf.keras.mixed_precision.set_global_policy("mixed_float16")

config = GPTConfig()

emb = GPTEmbeddings(config)


x = tf.random.normal((2,16,768))

attn = MultiHeadAttention(config)

y = attn(x)

print(y.shape)

input_ids = tf.random.uniform((2, 16), maxval=50257, dtype=tf.int32)
tf.keras.backend.clear_session()
model = Decoder(config)

logits = model(input_ids)

print(logits.shape)

X=tf.random.uniform((2,16),maxval=config.vocab_size,dtype=tf.int32)
loss_fn=GPTLoss()
opt=build_optimizer(config,opt="SGD")
loss=train_step(model,opt,loss_fn,X)

print(tf.shape(loss))

