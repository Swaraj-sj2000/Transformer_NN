#GPT/layers/mlp.py

import tensorflow as tf

class FFN(tf.keras.layers.Layer):
    def __init__(self,config):
        super().__init__()

        self.fc1=tf.keras.layers.Dense(4*config.n_embedd,activation='relu')
        self.fc2=tf.keras.layers.Dense(config.n_embedd)

    def call(self,X):
        return self.fc2(self.fc1(X))
    

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