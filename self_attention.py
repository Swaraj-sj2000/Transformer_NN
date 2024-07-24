import numpy as np
import math

E=np.array([[0.1,0.2,0.3,0.4],
           [0.5,0.6,0.7,0.8],
           [0.9,1.0,1.1,1.2],
           [1.3,1.4,1.5,1.6],
           [1.7,1.8,1.9,2.0],
           [2.1,2.2,2.3,2.4],
           [2.5,2.6,2.7,2.8],
           [2.9,3.0,3.1,3.2],
           [3.3,3.4,3.5,3.6],
           [3.7,3.8,3.9,4.0],
           [4.1,4.2,4.3,4.4]])

w_Q=np.array([[0.2,0.1,0.4,0.3],
              [0.3,0.4,0.1,0.2],
              [0.4,0.2,0.3,0.1],
              [0.1,0.3,0.2,0.4]])

w_K=np.array([[0.1,0.3,0.2,0.4],
              [0.4,0.2,0.3,0.1],
              [0.2,0.4,0.1,0.3],
              [0.3,0.1,0.4,0.2]])

w_V=np.array([[0.4,0.2,0.1,0.3],
              [0.3,0.4,0.2,0.1],
              [0.2,0.1,0.4,0.3],
              [0.1,0.3,0.2,0.4]])

d=4
Q=np.matmul(E,w_Q)
K=np.matmul(E,w_K)
V=np.matmul(E,w_V)
print(f"Q=\n{Q}\nK=\n{K}\nV={V}")

compatibility_mat=np.matmul(Q,K.T)
scale=(1/np.sqrt(d))*compatibility_mat

print(f"comp_mat={compatibility_mat}")
print(f"scale={scale}")

def rowwise_softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  
    sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)
    softmax_x = exp_x / sum_exp_x
    return softmax_x

attention_mat=rowwise_softmax(scale)

context_mat=np.matmul(attention_mat,V)

print(f"attention_mat=\n{attention_mat}\ncontext_mat=\n{context_mat}")