#GPT/layers/mlp.py

import tensorflow as tf

class FFN(tf.keras.layers.Layer):
    """Feed-forward network: Linear -> GELU -> Linear.
    
    Input: (batch, seq_len, n_embedd)
    Output: (batch, seq_len, n_embedd)
    
    Architecture:
    - Expand to 4x dimension
    - Apply GELU activation
    - Project back to original dimension
    """
    def __init__(self, config):
        super().__init__()
        self.c_fc = tf.keras.layers.Dense(config.d_ff)
        self.c_proj = tf.keras.layers.Dense(config.n_embedd)

    def call(self, X):
        X = self.c_fc(X)
        X = tf.nn.gelu(X, approximate=False)  
        X = self.c_proj(X)
        return X
    

class swiGLU(tf.keras.layers.Layer):
    def __init__(self,config):
        super().__init__()
       
        self.w1 = tf.keras.layers.Dense(4*config.n_embedd)
        self.w2 = tf.keras.layers.Dense(4*config.n_embedd)
        self.w3 = tf.keras.layers.Dense(config.n_embedd)

    def swish(self, x):
        return x * tf.nn.sigmoid(x)

    def call(self, x):
        return self.w3(self.w1(x) * self.swish(self.w2(x)))
