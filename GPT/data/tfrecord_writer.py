import tensorflow as tf
import os
from GPT.utils.logging import get_logger

logger=get_logger(__name__)

def shard_tfrecords(
    samples,
    config,
    num_shards=10,
    validation_split=0.2,
    prefix_train="train",
    prefix_val="val"
):
    logger.info("Writing TFRecords...")

    os.makedirs(config.train_dir, exist_ok=True)
    os.makedirs(config.val_dir,exist_ok=True)

    split_idx = int(len(samples) * (1 - validation_split))

    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    train_writers = [
        tf.io.TFRecordWriter(
            os.path.join(config.train_dir, f"{prefix_train}_{i:03d}.tfrecord")
        )
        for i in range(num_shards)
    ]

    val_writers = [
        tf.io.TFRecordWriter(
            os.path.join(config.val_dir, f"{prefix_val}_{i:03d}.tfrecord")
        )
        for i in range(max(1, num_shards // 5))
    ]

    def write(samples, writers):
        for idx, ids in enumerate(samples):
            shard_id = idx % len(writers)

            feature = {
                "input_ids": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=ids)
                )
            }

            example = tf.train.Example(
                features=tf.train.Features(feature=feature)
            )

            writers[shard_id].write(example.SerializeToString())

    write(train_samples, train_writers)
    write(val_samples, val_writers)

    for w in train_writers + val_writers:
        w.close()

    logger.info("Done writing TFRecords")

