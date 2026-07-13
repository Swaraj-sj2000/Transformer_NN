import numpy as np
import tensorflow as tf

from GPT.inference.sampler import Sampler


class DummyModel(tf.keras.Model):
    def call(self, inputs, training=False):
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        vocab_size = 4
        base = tf.range(vocab_size, dtype=tf.float32)
        logits = tf.repeat(base[tf.newaxis, tf.newaxis, :], repeats=batch_size, axis=0)
        logits = tf.repeat(logits, repeats=seq_len, axis=1)
        return logits


def test_generate_uses_common_decode_strategy():
    sampler = Sampler()
    model = DummyModel()
    input_ids = tf.constant([[1, 2]], dtype=tf.int32)

    greedy_output = sampler.generate(
        model,
        input_ids,
        max_new_tokens=2,
        decode_strategy="greedy",
    )
    sample_output = sampler.generate(
        model,
        input_ids,
        max_new_tokens=2,
        decode_strategy="sample",
    )

    assert greedy_output.shape == (1, 4)
    assert sample_output.shape == (1, 4)
    assert np.array_equal(greedy_output.numpy()[0, :2], input_ids.numpy()[0])
