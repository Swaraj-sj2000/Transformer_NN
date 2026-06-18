#!/bin/python
#GPT/causal_Attention.py
"""
=====================================================
File: causal_attention.py

Purpose:
Implements masked multi-head self-attention.

Input:
    X : (B,T,C)

Output:
    Attention output : (B,T,C)

Tensor Flow:

(B,T,C)
    ↓
WQ WK WV
    ↓
(B,H,T,D)
    ↓
QKᵀ / √D
    ↓
(B,H,T,T)
    ↓
Causal Mask
    ↓
Softmax
    ↓
Attention weights
    ↓
× V
    ↓
(B,H,T,D)
    ↓
Merge heads
    ↓
(B,T,C)
    ↓
WO
    ↓
(B,T,C)

Used by:
    Transformer Block
=====================================================
"""
import tensorflow as tf

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self,config):
        super().__init__()

        self.h=config.n_head
        self.d_model=config.n_embedd
        self.d_k = self.d_model//self.h

        assert config.n_embedd%config.n_head==0
        self.c_attn = tf.keras.layers.Dense(3*self.d_model)
        self.WO=tf.keras.layers.Dense(self.d_model)

    def call(self,X):
        shape = tf.shape(X)

        batch_size = shape[0]
        seq_len = shape[1]
        d_model = shape[2]        
        h=self.h
        d_k=self.d_k

        c_attn=self.c_attn(X)
        Q,K,V=tf.split(c_attn,3,axis=-1)

        Q_h=tf.transpose(tf.reshape(Q,shape=(batch_size,seq_len,h,d_k)),perm=[0,2,1,3])
        K_h=tf.transpose(tf.reshape(K,shape=(batch_size,seq_len,h,d_k)),perm=[0,2,1,3])
        V_h=tf.transpose(tf.reshape(V,shape=(batch_size,seq_len,h,d_k)),perm=[0,2,1,3])

        z_score=tf.matmul(Q_h,K_h,transpose_b=True)

        scaled_z_score=z_score/tf.sqrt(tf.cast(d_k,tf.float32))

        mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)),-1,0)
        causal_masked_score=scaled_z_score+mask[tf.newaxis,tf.newaxis,:,:]*(-1e9)
        
        QK_logits=tf.nn.softmax(scaled_z_score+causal_masked_score,axis=-1)
        attention_head_score=tf.matmul(QK_logits,V_h)
        attention_score=tf.reshape(tf.transpose(attention_head_score,perm=[0,2,1,3]),shape=(batch_size,seq_len,d_model))
        return self.WO(attention_score)







        

