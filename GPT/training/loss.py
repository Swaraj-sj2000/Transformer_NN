#GPT/training/loss.py
import tensorflow as tf


class GPTLoss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()

        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction="none"
        )

    def call(self, logits, targets):

        B, T, V = tf.shape(logits)[0], tf.shape(logits)[1], tf.shape(logits)[2]

        logits = tf.reshape(logits, (B * T, V))
        targets = tf.reshape(targets, (B * T,))

        loss = self.loss_fn(targets, logits)

        return tf.reduce_mean(loss)