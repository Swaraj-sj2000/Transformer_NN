#GPT/model/gpt.py
import tensorflow as tf

from layers.embedding import GPTEmbeddings
from layers.block import DecoderBlock

class Decoder(tf.keras.layers.Layer):
    def __init__(self,config):
        super().__init__()
        self.embedding=GPTEmbeddings(config)
        self.decoder_layers=[DecoderBlock(config) for _ in range(config.n_layer)]
        self.ln_f = tf.keras.layers.LayerNormalization(epsilon=1e-5)

        self.final_linear=tf.keras.layers.Dense(config.vocab_size)

    
    def call(self, input_ids):

        x = self.embedding(input_ids)

        for block in self.decoder_layers:
            x = block(x)

        x = self.ln_f(x)

        logits = self.final_linear(x)

        return logits


