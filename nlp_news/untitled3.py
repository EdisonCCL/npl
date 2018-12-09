# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 15:53:07 2018

@author: j
"""
import tensorflow as tf
import numpy as np
tf.reset_default_graph()
X=np.random.randn(2,4,5)
X[1,1:]=0
seq_lengths=[4,1]
cell=tf.contrib.rnn.BasicLSTMCell(num_units=5,state_is_tuple=True)
gru=tf.contrib.rnn.GRUCell(3)
outputs,last_states=tf.nn.dynamic_rnn(cell,X,seq_lengths,dtype=tf.float64)
gruoutputs,grulast_states=tf.nn.dynamic_rnn(gru,X,seq_lengths,dtype=tf.float64)
sess=tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
result,sta,gruout,grusta=sess.run([outputs,last_states,gruoutputs,grulast_states])
print('全序列：\n',result[0])
print('短序列：\n',result[1])
print('LSTM的状态：',len(sta),'\n',sta[1])
print('GRU的短序列：\n',gruout[1])
print('GRU的状态：',len(grusta),'\n',grusta[1])
sess.close()