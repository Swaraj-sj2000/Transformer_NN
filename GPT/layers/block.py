#GPT/layers/block.py

import tensorflow as tf

from GPT.layers.mlp import FFN
from GPT.layers.attention import MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """Transformer decoder block: LayerNorm -> Attention -> Residual -> LayerNorm -> FFN -> Residual.
    
    Pre-norm residual connections for training stability.
    """
    def __init__(self, config):
        super().__init__()
        self.ln_1 = tf.keras.layers.LayerNormalization(epsilon=config.epsilon)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = tf.keras.layers.LayerNormalization(epsilon=config.epsilon)
        self.mlp = FFN(config)

    def call(self, X):
        # Attention with pre-norm residual
        attn_out = self.attn(self.ln_1(X))
        X = X + attn_out
        
        # FFN with pre-norm residual
        mlp_out = self.mlp(self.ln_2(X))
        X = X + mlp_out
        
        return X


    
