#GPT/tfrecords.py

import tensorflow as tf
import os
import requests
import tiktoken

from config import GPTConfig

def shard_tfrecords(
    samples,
    config,
    num_shards=10,
    validation_split=0.2,
    prefix_train="train",
    prefix_val="val"
):
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

def parse_fn(example_proto,block_size):
    feature_description={
        "input_ids":tf.io.FixedLenFeature([block_size],tf.int64),
    }

    example=tf.io.parse_single_example(example_proto,feature_description)
    x=tf.cast(example['input_ids'],tf.int32)
    return x

def get_dataset(files,batch_size,block_size):
    ds=tf.data.TFRecordDataset(files,
                               num_parallel_reads=tf.data.AUTOTUNE)
    ds = (ds.map(lambda x:parse_fn(x,block_size), 
             num_parallel_calls=tf.data.AUTOTUNE))
    


    ds=(ds.shuffle(10000)
        .repeat()
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE))
      
    return ds


def build_samples(text,block_size):
    enc=tiktoken.get_encoding("gpt2")

    tokens=enc.encode(text)
    samples=[]
    stride=block_size

    for i in range(0,len(tokens)-block_size,stride):
        chunk=tokens[i:i+block_size]
        samples.append(chunk)

    return samples

if __name__ == "__main__":
    config=GPTConfig()

    os.makedirs("data", exist_ok=True)

    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

    path = "GPT/data/shakespeare.txt"

    if not os.path.exists(path):
        r=requests.get(url)
        r.raise_for_status()
        with open(path, "w",encoding='utf-8') as f:
            f.write(r.text)
        print("Downloaded Shakespeare dataset....")

    else: print("Dataset already exists...")

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    print("Tokenizing...") 
    samples = build_samples(text, config.block_size)

    print("Total samples:", len(samples))
    print("Train samples:", int(0.8 * len(samples)))
    print("Val samples:", len(samples) - int(0.8 * len(samples)))

    print("Writing TFRecords...")
    shard_tfrecords(
        samples,
        config,
        num_shards=10,
        validation_split=0.2
    )

    print("Done writing TFRecords")
    