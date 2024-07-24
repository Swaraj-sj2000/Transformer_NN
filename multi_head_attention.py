import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
import matplotlib.pyplot as plt

seq_Length = 4    # length of the input sequence
batch_Size = 1
input_Dim = 512   # size of each encoded vector for each word
d_Model = 512     # size of output vector for each word

# Generate random input
x = tf.random.normal((batch_Size, seq_Length, input_Dim))
print("Input shape:", x.shape)

# Linear layer to generate Q, K, V
qkv_layer = layers.Dense(3 * d_Model, input_shape=(input_Dim,))
qkv = qkv_layer(x)
print("QKV shape:", qkv.shape)

# Define number of heads and head dimension
num_Heads = 8
head_Dim = d_Model // num_Heads

# Reshape QKV for multi-head attention
qkv = tf.reshape(qkv, (batch_Size, seq_Length, num_Heads, 3 * head_Dim))
print("Reshaped QKV shape:", qkv.shape)

# Transpose to bring heads to second dimension
qkv = tf.transpose(qkv, perm=[0, 2, 1, 3])
print("Transposed QKV shape:", qkv.shape)

# Split QKV into Q, K, V
q, k, v = tf.split(qkv, num_or_size_splits=3, axis=-1)
print("Q shape:", q.shape)
print("K shape:", k.shape)
print("V shape:", v.shape)

# Scaled dot-product attention
d_k = tf.cast(tf.shape(q)[-1], tf.float32)
scaled = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(d_k)
print("Scaled attention logits shape:", scaled.shape)

# Create mask
mask = tf.fill(scaled.shape, float('-inf'))
triu_mask = tf.linalg.band_part(tf.ones((batch_Size, num_Heads, seq_Length, seq_Length)), num_lower=0, num_upper=-1)
diagonal = tf.zeros((batch_Size, num_Heads, seq_Length))
triu_mask = tf.linalg.set_diag(triu_mask, diagonal)
mask = tf.where(triu_mask == 1, mask, tf.zeros_like(mask))
scaled += mask

# Softmax to get attention weights
attention = tf.nn.softmax(scaled, axis=-1)
print("Attention weights shape:", attention.shape)

# Calculate the attention output
values = tf.matmul(attention, v)
print("Attention output shape:", values.shape)

# Combine heads and project
values = tf.transpose(values, perm=[0, 2, 1, 3])
combined_values = tf.reshape(values, (batch_Size, seq_Length, d_Model))

# Final linear layer to project combined values
output_layer = layers.Dense(d_Model)
output = output_layer(combined_values)
print("Output shape:", output.shape)
