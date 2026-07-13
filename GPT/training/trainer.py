#GPT/training/trainer.py

import tensorflow as tf
import tqdm

from GPT.config import GPTConfig
from GPT.training.loss import GPTLoss
from GPT.training.optimizer import build_optimizer
from GPT.model.gpt import Decoder
from GPT.data.build_dataset_pipeline import get_dataset

config=GPTConfig()

clip_norm=config.grad_clip_norm
accum_steps=config.accum_steps

logger=TrainLogger()
loss_fn=GPTLoss()



@tf.function
def train_step(model, optimizer, loss_fn,x,clip_norm):
    try:
        with tf.GradientTape() as tape:
            logits = model(x)
            targets = x[:, 1:]
            logits = logits[:, :-1, :]

            loss = loss_fn(logits, targets)
            #loss=tf.reduce_mean(loss)

        grads = tape.gradient(loss, model.trainable_variables)
        grad_norm=tf.linalg.global_norm(grads)

        grads,_=tf.clip_by_global_norm(grads,clip_norm)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return loss,grad_norm
    except Exception as e:
        logger.log(
            status="error",
            error=str(e)
        )
        raise



if __name__=="__main__":
    model=Decoder(config=config)
    optimizer=build_optimizer(config,"SGD")
    train_files = tf.data.Dataset.list_files(config.train_dir + "/*.tfrecord")
    val_files = tf.data.Dataset.list_files(config.val_dir + "/*.tfrecord")
    train_files = [f.numpy().decode() for f in train_files]
    val_files = [f.numpy().decode() for f in val_files]

    train_ds=get_dataset(train_files,batch_size=config.batch_size,block_size=config.block_size)
    val_ds=get_dataset(val_files,batch_size=config.batch_size,block_size=config.block_size)


    for step, batch in enumerate(tqdm.tqdm(train_ds)):

        loss, grad_norm = train_step(
            model,
            optimizer,
            loss_fn,
            batch,
            clip_norm
        )

        if step % 50 == 0:
            print("Step:", step, "loss:", loss.numpy(), "grad_norm:", grad_norm.numpy())

        if step % 200 == 0:
            val_loss = 0.0
            n = 0

            for vbatch in val_ds:
                logits = model(vbatch)
                targets = vbatch[:, 1:]
                logits = logits[:, :-1, :]

                loss = loss_fn(logits, targets)
                val_loss += tf.reduce_mean(loss)
                n += 1

            print("VAL LOSS:", (val_loss / n).numpy())