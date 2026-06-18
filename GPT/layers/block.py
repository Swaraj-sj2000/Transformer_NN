#GPT/layers/block.py

import tensorflow as tf

from layers.mlp import FFN
from layers.attention import MultiHeadAttention

class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self,config):
        super().__init__()

        self.LN1=tf.keras.layers.LayerNormalization(epsilon=1e-9)
        self.MHA=MultiHeadAttention(config)
        self.LN2=tf.keras.layers.LayerNormalization(epsilon=1e-9)
        self.MLP=FFN(config)

    def call(self,X):
        attn=self.MHA(X)
        Z1=self.LN1(X+attn)
        ffn_op=self.MLP(Z1)
        return self.LN2(Z1+ffn_op)

