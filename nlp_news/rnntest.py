# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 10:20:36 2018

@author: ccl
"""

with open('../mynew/test.txt', encoding='gb18030') as file:
    line_list = [k.strip() for k in file.readlines()]
    test_label_list = [k.split()[0] for k in line_list]
    test_content_list = [k.split(maxsplit=1)[1] for k in line_list]
with open('../mynew/vocabulary.txt', encoding='gb18030') as file:
    vocabulary_list = [k.strip() for k in file.readlines()]
print('0.load test data finished')

word2id_dict = dict([(b, a) for a, b in enumerate(vocabulary_list)])
content2idList = lambda content : [word2id_dict[word] for word in content if word in word2id_dict]
test_idlist_list = [content2idList(content) for content in test_content_list]
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
test_X = kr.preprocessing.sequence.pad_sequences(test_idlist_list, sequence_length)
from sklearn.preprocessing import LabelEncoder
labelEncoder = LabelEncoder()
test_y = labelEncoder.fit_transform(test_label_list)
test_Y = kr.utils.to_categorical(test_y, num_classes)
import tensorflow as tf
tf.reset_default_graph()
X_holder = tf.placeholder(tf.int32, [None, sequence_length])
Y_holder = tf.placeholder(tf.float32, [None, num_classes])
print('1.data preparation finished')

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

isCorrect = tf.equal(tf.argmax(Y_holder,1), tf.argmax(predict_Y, 1))
accuracy = tf.reduce_mean(tf.cast(isCorrect, tf.float32))
print('2.build model finished')

saver=tf.train.Saver()
init = tf.global_variables_initializer()
with tf.Session() as sess2:
    sess2.run(init)
    saver.restore(sess2,'D:/nlp_news/log/rnnmodel.cpkt')
    predict_value,accuracy_value = sess2.run([predict_Y,accuracy], {X_holder:test_X, Y_holder:test_Y})
 
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

y = np.argmax(predict_value, axis=1)
predict_label_list = labelEncoder.inverse_transform(y)
result=pd.DataFrame(confusion_matrix(test_label_list, predict_label_list), 
             columns=labelEncoder.classes_,
             index=labelEncoder.classes_ )
print(result)
print('3.test confusion matrix finished')
from sklearn.metrics import precision_recall_fscore_support

def eval_model(y_true, y_pred, labels):
    # 计算每个分类的Precision, Recall, f1, support
    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred)
    # 计算总体的平均Precision, Recall, f1, support
    tot_p = np.average(p, weights=s)
    tot_r = np.average(r, weights=s)
    tot_f1 = np.average(f1, weights=s)
    tot_s = np.sum(s)
    res1 = pd.DataFrame({
        u'Label': labels,
        u'Precision': p,
        u'Recall': r,
        u'F1': f1,
        u'Support': s
    })
    res2 = pd.DataFrame({
        u'Label': ['总体'],
        u'Precision': [tot_p],
        u'Recall': [tot_r],
        u'F1': [tot_f1],
        u'Support': [tot_s]
    })
    res2.index = [8]
    res = pd.concat([res1, res2])
    return res[['Label', 'Precision', 'Recall', 'F1', 'Support']]

result1=eval_model(test_label_list, predict_label_list, labelEncoder.classes_)
print(result1)
#print('4.test evaluation finished')
print('accuracy:%.4f' %(accuracy_value))
print('5.test model finished')