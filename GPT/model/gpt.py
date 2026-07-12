#GPT/model/gpt.py
import tensorflow as tf

from tensorflow.keras import mixed_precision

from GPT.layers.embedding import GPTEmbeddings
from GPT.layers.block import DecoderBlock

mixed_precision.set_global_policy("mixed_float16")

class Decoder(tf.keras.Model):
    """Full GPT-2 model: Embeddings -> Transformer blocks -> Output projection.
    
    Input: (batch, seq_len) token IDs
    Output: (batch, seq_len, vocab_size) logits
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.embedding = GPTEmbeddings(config)
        self.decoder_layers = [DecoderBlock(config) for _ in range(config.n_layer)]
        self.ln_f = tf.keras.layers.LayerNormalization(epsilon=config.epsilon)

    def call(self, input_ids, return_hidden: bool = False):
        """Forward pass.
        
        Args:
            input_ids: (batch, seq_len) token IDs
            return_hidden: If True, return dict of all intermediate activations
        
        Returns:
            logits: (batch, seq_len, vocab_size)
            or dict of activations if return_hidden=True
        """
        activations = {}

        # Embeddings
        x = self.embedding(input_ids)
        activations['embedding'] = x

        # Transformer blocks
        for i, block in enumerate(self.decoder_layers):
            x = block(x)
            activations[f'block_{i}'] = x

        # Final layer norm
        x = self.ln_f(x)
        activations['ln_f'] = x
        
        # Project to vocab
        logits = tf.matmul(x, self.embedding.wte.embeddings, transpose_b=True)
        activations['logits'] = logits

        if return_hidden:
            return activations
        return logits
