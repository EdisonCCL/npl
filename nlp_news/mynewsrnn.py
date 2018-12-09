# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 14:30:55 2018

@author: j
"""

import warnings
warnings.filterwarnings('ignore')
import time
startTime = time.time()
def printUsedTime():
    used_time = time.time() - startTime
    print('used time: %.2f seconds' %used_time)
with open('H:/Anaconda/cnews/cnews.train.txt', encoding='utf8') as file:
    line_list = [k.strip() for k in file.readlines()]
    train_label_list = [k.split()[0] for k in line_list]
    train_content_list = [k.split(maxsplit=1)[1] for k in line_list]
with open('H:/Anaconda/cnews/cnews.vocab.txt', encoding='utf8') as file:
    vocabulary_list = [k.strip() for k in file.readlines()]
print('0.load train data finished')
printUsedTime()

word2id_dict = dict([(b, a) for a, b in enumerate(vocabulary_list)])
content2idList = lambda content : [word2id_dict[word] for word in content if word in word2id_dict]
train_idlist_list = [content2idList(content) for content in train_content_list]
vocabolary_size = 5000  # 词汇表大小
sequence_length = 150  # 序列长度
embedding_size = 64  # 词向量大小
num_hidden_units = 256  # LSTM细胞隐藏层大小
num_fc1_units = 64 #第1个全连接下一层的大小
dropout_keep_probability = 0.5  # dropout保留比例
num_classes = 10  # 类别数量
learning_rate = 1e-3  # 学习率
batch_size = 64  # 每批训练大小
import tensorflow.contrib.keras as kr
train_X = kr.preprocessing.sequence.pad_sequences(train_idlist_list, sequence_length)
from sklearn.preprocessing import LabelEncoder
labelEncoder = LabelEncoder()
train_y = labelEncoder.fit_transform(train_label_list)
train_Y = kr.utils.to_categorical(train_y, num_classes)
import tensorflow as tf
tf.reset_default_graph()
X_holder = tf.placeholder(tf.int32, [None, sequence_length])
Y_holder = tf.placeholder(tf.float32, [None, num_classes])
print('1.data preparation finished')
printUsedTime()

embedding = tf.get_variable('embedding', 
                            [vocabolary_size, embedding_size])
embedding_inputs = tf.nn.embedding_lookup(embedding, 
                                          X_holder)
gru_cell = tf.contrib.rnn.GRUCell(num_hidden_units)
outputs, state = tf.nn.dynamic_rnn(gru_cell,
                                   embedding_inputs, 
                                   dtype=tf.float32)
last_cell = outputs[:, -1, :]
full_connect1 = tf.layers.dense(last_cell,
                                num_fc1_units)
full_connect1_dropout = tf.contrib.layers.dropout(full_connect1,
                                                  dropout_keep_probability)
full_connect1_activate = tf.nn.relu(full_connect1_dropout)
full_connect2 = tf.layers.dense(full_connect1_activate,
                                num_classes)
predict_Y = tf.nn.softmax(full_connect2)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y_holder,
                                                          logits=full_connect2)
loss = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)
isCorrect = tf.equal(tf.argmax(Y_holder,1), tf.argmax(predict_Y, 1))
accuracy = tf.reduce_mean(tf.cast(isCorrect, tf.float32))
print('2.build model finished')
printUsedTime()

init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)
print('3.initialize variable finished')
printUsedTime()

with open('H:/Anaconda/cnews/cnews.test.txt', encoding='utf8') as file:
    line_list = [k.strip() for k in file.readlines()]
    test_label_list = [k.split()[0] for k in line_list]
    test_content_list = [k.split(maxsplit=1)[1] for k in line_list]
test_idlist_list = [content2idList(content) for content in test_content_list]
test_X = kr.preprocessing.sequence.pad_sequences(test_idlist_list, sequence_length)
test_y = labelEncoder.transform(test_label_list)
test_Y = kr.utils.to_categorical(test_y, num_classes)
print('4.load test data finished')
printUsedTime()

print('5.begin model training')
import random
for i in range(5000):
    selected_index = random.sample(list(range(len(train_y))), k=batch_size)
    batch_X = train_X[selected_index]
    batch_Y = train_Y[selected_index]
    session.run(train, {X_holder:batch_X, Y_holder:batch_Y})
    step = i + 1 
    if step % 100 == 0:
        selected_index = random.sample(list(range(len(test_y))), k=200)
        batch_X = test_X[selected_index]
        batch_Y = test_Y[selected_index]
        loss_value, accuracy_value = session.run([loss, accuracy], {X_holder:batch_X, Y_holder:batch_Y})
        print('step:%d loss:%.4f accuracy:%.4f' %(step, loss_value, accuracy_value))
        printUsedTime()
