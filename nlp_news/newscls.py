# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 21:58:42 2018

@author: j
"""
def predict(newstest):
  import os
  with open('H:/Anaconda/cnews/mynew/vocabulary.txt','r', encoding='gb18030') as file:
    vocabulary_list = [k.strip() for k in file.readlines()]
  word2id_dict = dict([(b, a) for a, b in enumerate(vocabulary_list)])
  content2idList = lambda content : [word2id_dict[word] for word in content if word in word2id_dict]
  vocab_size = 5000  # 词汇表大小
  seq_length = 600  # 序列长度
  embedding_dim = 64  # 词向量维度
  num_classes = 8  # 类别数
  num_filters = 256  # 卷积核数目
  kernel_size = 5  # 卷积核尺寸
  hidden_dim = 128  # 全连接层神经元

  import tensorflow.contrib.keras as kr

  import tensorflow as tf
  tf.reset_default_graph()
  X_holder = tf.placeholder(tf.int32, [None, seq_length])

  embedding = tf.get_variable('embedding', [vocab_size, embedding_dim])
  embedding_inputs = tf.nn.embedding_lookup(embedding, X_holder)
  conv = tf.layers.conv1d(embedding_inputs, num_filters, kernel_size)
  max_pooling = tf.reduce_max(conv, reduction_indices=[1])
  full_connect = tf.layers.dense(max_pooling, hidden_dim)
  full_connect_dropout = tf.contrib.layers.dropout(full_connect, keep_prob=0.75)
  full_connect_activate = tf.nn.relu(full_connect_dropout)
  softmax_before = tf.layers.dense(full_connect_activate, num_classes)
  predict_Y = tf.nn.softmax(softmax_before)

  saver=tf.train.Saver()
  init = tf.global_variables_initializer()
  test_idlist_list = [content2idList(newstest)] 
  test_X = kr.preprocessing.sequence.pad_sequences(test_idlist_list, seq_length)
  id2label={0:'auto',1:'business', 2:'house',3:'it', 5:'travel', 6:'women', 4:'sports', 7:'yule'}
  with tf.Session() as sess2:
    sess2.run(init)
    saver.restore(sess2,'H:/Anaconda/nlp_news/log/linermodel.cpkt')
    label=sess2.run(tf.argmax(predict_Y, 1),feed_dict={X_holder:test_X})
  print('The news is about %s.' %id2label[int(label)])

