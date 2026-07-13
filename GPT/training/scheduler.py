#GPT/training/scheduler.py
import tensorflow as tf
import math


class CosineWarmupSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, config):
        super().__init__()

        self.base_lr = config.learning_rate
        self.warmup_steps = tf.cast(config.warmup_steps, tf.float32)
        self.total_steps = tf.cast(config.total_steps, tf.float32)

    def __call__(self, step):

        step = tf.cast(step, tf.float32)

        warmup_steps = tf.maximum(self.warmup_steps, 1.0)
        total_steps = tf.maximum(self.total_steps, self.warmup_steps + 1.0)


        warmup_lr = self.base_lr * (step / warmup_steps)

   
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        progress = tf.clip_by_value(progress, 0.0, 1.0)

        cosine_lr = 0.5 * self.base_lr * (1 + tf.cos(math.pi * progress))

        return tf.cond(
            step < warmup_steps,
            lambda: warmup_lr,
            lambda: cosine_lr
        )
    

    