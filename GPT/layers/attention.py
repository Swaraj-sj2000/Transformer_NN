#GPT/layers/attention.py
"""
=====================================================
File: causal_attention.py

Purpose:
Implements masked multi-head self-attention.

Input:
    X : (B,T,C)

Output:
    Attention output : (B,T,C)

Tensor Flow:

(B,T,C)
    ↓
WQ WK WV
    ↓
(B,H,T,D)
    ↓
QKᵀ / √D
    ↓
(B,H,T,T)
    ↓
Causal Mask
    ↓
Softmax
    ↓
Attention weights
    ↓
× V
    ↓
(B,H,T,D)
    ↓
Merge heads
    ↓
(B,T,C)
    ↓
WO
    ↓
(B,T,C)

Used by:
    Transformer Block
=====================================================
"""
import tensorflow as tf

class MultiHeadAttention(tf.keras.layers.Layer):
    """Multi-head self-attention with causal masking.
    
    Input: (batch, seq_len, n_embedd)
    Output: (batch, seq_len, n_embedd)
    
    Process:
    1. Project to Q, K, V
    2. Split into multiple heads
    3. Compute attention scores with causal mask
    4. Apply softmax and aggregate
    5. Project output
    """
    def __init__(self, config):
        super().__init__()
        self.h = config.n_head
        self.d_model = config.n_embedd
        self.d_k = config.n_embedd // config.n_head

        assert config.n_embedd % config.n_head == 0, "n_embedd must be divisible by n_head"
        
        self.c_attn = tf.keras.layers.Dense(3 * config.n_embedd)  # Combined QKV projection
        self.c_proj = tf.keras.layers.Dense(config.n_embedd)      # Output projection

    def _create_causal_mask(self, seq_len):
        """Create causal attention mask: 0 where we attend, -1e4 where we don't."""
        lower_tri = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        mask = 1.0 - lower_tri
        return mask[tf.newaxis, tf.newaxis, :, :]

    def call(self, X):
        B = tf.shape(X)[0]
        T = tf.shape(X)[1]
        D = self.d_model
        h = self.h
        dk = self.d_k

        # Project and split into heads
        qkv = self.c_attn(X)
        Q, K, V = tf.split(qkv, 3, axis=-1)
        
        Q = tf.reshape(Q, [B, T, h, dk])
        K = tf.reshape(K, [B, T, h, dk])
        V = tf.reshape(V, [B, T, h, dk])
        
        Q = tf.transpose(Q, [0, 2, 1, 3])  # (B, h, T, dk)
        K = tf.transpose(K, [0, 2, 1, 3])
        V = tf.transpose(V, [0, 2, 1, 3])

        # Compute attention scores
        scores = tf.matmul(Q, tf.transpose(K, [0, 1, 3, 2]))
        scores = scores / tf.sqrt(tf.cast(dk, scores.dtype))

        # Apply causal mask
        mask = self._create_causal_mask(T)
        scores = scores + mask * tf.cast(-1e4, scores.dtype)

        # Softmax and aggregate
        weights = tf.nn.softmax(scores, axis=-1)
        context = tf.matmul(weights, V)  # (B, h, T, dk)

        # Reshape and project
        context = tf.transpose(context, [0, 2, 1, 3])  # (B, T, h, dk)
        context = tf.reshape(context, [B, T, D])

        return self.c_proj(context)



if __name__=="__main__":
    from GPT.config import GPTConfig
    config=GPTConfig()
    mha=MultiHeadAttention(config)
    X=tf.random.normal((2,4,8))
    print(mha(X).shape)
