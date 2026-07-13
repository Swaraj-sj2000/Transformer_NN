#GPT/training/optimizer.py
import tensorflow as tf
from GPT.training.scheduler import CosineWarmupSchedule

def build_optimizer(config,opt="adam"):
    lr_sh=CosineWarmupSchedule(config)

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=lr_sh,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8
    )
    if opt=="SGD":
        optimizer=tf.keras.optimizers.SGD(
            learning_rate=lr_sh,

        )

    elif opt=="AdaGrad":
        optimizer=tf.keras.optimizers.Adagrad(
            learning_rate=lr_sh,
            initial_accumulator_value=0.1,

        )


    elif opt=="RMSProp":
        optimizer=tf.keras.optimizers.RMSprop(
                    learning_rate=lr_sh,
                    rho=0.9,
                    momentum=0.0,
                    epsilon=1e-07,
                    ema_momentum=0.99,
                    )
        
    
    elif opt=="Lion":
        optimizer=tf.keras.optimizers.Lion(
                    learning_rate=lr_sh,
                    beta_1=0.9,
                    beta_2=0.99,
                    ema_momentum=0.99,
                
                )

    elif opt=="Adafactor":
        optimizer=tf.keras.optimizers.Adafactor(
                    learning_rate=lr_sh,
                    beta_2_decay=-0.8,
                    epsilon_1=1e-30,
                    epsilon_2=0.001,
                    clip_threshold=1.0,
                    ema_momentum=0.99,)
                

    elif opt=="ZeRO":
        pass

    return optimizer