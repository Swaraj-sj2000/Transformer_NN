import tensorflow as tf
import os
from GPT.utils.logging import get_logger

logger=get_logger(__name__)

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
