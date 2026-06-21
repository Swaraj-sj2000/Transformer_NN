import tensorflow as tf

def checkpoint_block(block):

    def forward(x):
        def inner(x):
            return block(x)
        
        @tf.custom_gradient
        def fn(x):
            y=inner(x)

            def grad(dy):
                with tf.GradientTape() as tape:
                    tape.watch(x)
                    y=inner(x)

                dx=tape.gradient(y,x,output_gradients=dy)
                return dx
            
            return y,grad
        
        return fn(x)
    
    return forward
        