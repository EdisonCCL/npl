# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 09:39:14 2018

@author: j
"""
import os
with open('H:/Anaconda/cnews/mynew/train.txt', encoding='gb18030') as file:
    line_list = [k.strip() for k in file.readlines()]
    train_label_list = [k.split()[0] for k in line_list]
    train_content_list = [k.split(maxsplit=1)[1] for k in line_list]
with open('H:/Anaconda/cnews/mynew/vocabulary.txt', encoding='gb18030') as file:
    vocabulary_list = [k.strip() for k in file.readlines()]
word2id_dict = dict([(b, a) for a, b in enumerate(vocabulary_list)])
content2idList = lambda content : [word2id_dict[word] for word in content if word in word2id_dict]
train_idlist_list = [content2idList(content) for content in train_content_list]
vocab_size = 5000  # 词汇表大小
seq_length = 600  # 序列长度
embedding_dim = 64  # 词向量维度
num_classes = 8  # 类别数
num_filters = 256  # 卷积核数目
kernel_size = 5  # 卷积核尺寸
hidden_dim = 128  # 全连接层神经元
dropout_keep_prob = 0.5  # dropout保留比例
learning_rate = 1e-3  # 学习率
batch_size = 64  # 每批训练大小

import tensorflow.contrib.keras as kr
train_X = kr.preprocessing.sequence.pad_sequences(train_idlist_list, seq_length)#40000*600的ndarray
from sklearn.preprocessing import LabelEncoder
labelEncoder = LabelEncoder()
train_y = labelEncoder.fit_transform(train_label_list)#1*40000的ndarray
train_Y = kr.utils.to_categorical(train_y, num_classes)#40000*8的ndarray
import tensorflow as tf
tf.reset_default_graph()
X_holder = tf.placeholder(tf.int32, [None, seq_length])
Y_holder = tf.placeholder(tf.float32, [None, num_classes])
embedding = tf.get_variable('embedding', [vocab_size, embedding_dim])#5000,64
embedding_inputs = tf.nn.embedding_lookup(embedding, X_holder)#?,600,64 输入层以后的结果
conv = tf.layers.conv1d(embedding_inputs, num_filters, kernel_size)#?,596,256 卷积层以后的结果
max_pooling = tf.reduce_max(conv, reduction_indices=[1])#?,256 池化层以后的结果
full_connect = tf.layers.dense(max_pooling, hidden_dim)#?,128 全连接层
full_connect_dropout = tf.contrib.layers.dropout(full_connect, keep_prob=0.75)#?,128 防止过拟合
full_connect_activate = tf.nn.relu(full_connect_dropout)#?,128 用激活函数
softmax_before = tf.layers.dense(full_connect_activate, num_classes)#?,8 全连接层
predict_Y = tf.nn.softmax(softmax_before)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y_holder, logits=softmax_before)
loss = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)
isCorrect = tf.equal(tf.argmax(Y_holder, 1), tf.argmax(predict_Y, 1))
accuracy = tf.reduce_mean(tf.cast(isCorrect, tf.float32))

saver=tf.train.Saver()

init = tf.global_variables_initializer()


with open('H:/Anaconda/cnews/mynew/test.txt', encoding='gb18030') as file:
    line_list = [k.strip() for k in file.readlines()]
    test_label_list = [k.split()[0] for k in line_list]
    test_content_list = [k.split(maxsplit=1)[1] for k in line_list]
test_idlist_list = [content2idList(content) for content in test_content_list]
test_X = kr.preprocessing.sequence.pad_sequences(test_idlist_list, seq_length)
test_y = labelEncoder.transform(test_label_list)
test_Y = kr.utils.to_categorical(test_y, num_classes)
import random
with tf.Session() as sess:
  sess.run(init)
  for i in range(600):
    selected_index = random.sample(list(range(len(train_y))), k=batch_size)
    batch_X = train_X[selected_index]
    batch_Y = train_Y[selected_index]
    sess.run(train, {X_holder:batch_X, Y_holder:batch_Y})
    step = i + 1 
    if step % 60 == 0:
        selected_index = random.sample(list(range(len(test_y))), k=200)
        batch_X = test_X[selected_index]
        batch_Y = test_Y[selected_index]
        loss_value, accuracy_value = sess.run([loss, accuracy], {X_holder:batch_X, Y_holder:batch_Y})
        print('step:%d loss:%.4f accuracy:%.4f' %(step, loss_value, accuracy_value))
  saver.save(sess,'H:/Anaconda/nlp_news/log/'+'linermodel.cpkt')
