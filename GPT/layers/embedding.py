#GPT/layers/embedding.py

"""
=====================================================
File: embedding.py

Purpose:
Implements GPT token and positional embeddings.

Input:
    input_ids : (B,T)

Output:
    embeddings : (B,T,n_embedd)

Components:
    - Token embedding (WTE)
    - Position embedding (WPE)

Used by:
    GPT model

Tensor flow:
    (B,T)
      ↓
    WTE
      ↓
    (B,T,C)

    +
    WPE
      ↓

    (B,T,C)
=====================================================
"""
import tensorflow as tf
class GPTEmbeddings(tf.keras.layers.Layer):
    """Token and positional embeddings.
    
    Input: (batch, seq_len) token IDs
    Output: (batch, seq_len, n_embedd) embeddings
    """
    def __init__(self, config):
        super().__init__()
        self.wte = tf.keras.layers.Embedding(config.vocab_size, config.n_embedd)  # Token embeddings
        self.wpe = tf.keras.layers.Embedding(config.block_size, config.n_embedd)   # Position embeddings

    def call(self, input_ids):
        T = tf.shape(input_ids)[1]
        pos = tf.range(T)[tf.newaxis, :]
        tok_emb = self.wte(input_ids)
        pos_emb = self.wpe(pos)
        return tok_emb + pos_emb



