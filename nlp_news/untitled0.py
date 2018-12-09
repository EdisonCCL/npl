# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 08:27:13 2018

@author: j
"""

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
x='【ＩＴ１６８　新闻资讯】“迅雷律师信不讲信用！”超级兔子作者蔡旋６月２４日愤怒地告诉ＩＴ１６８记者，四月初收到迅雷的律师函之后，他已经按照信中的要求删除所有“迅雷（超级兔子版）软件”，但还是被迅雷告上法庭。据悉，该案将于７月９日再次开庭。整个事件的起源还是来自蔡旋开发的一款软件《迅雷超级兔子版》，该软件推出后让迅雷感觉很受伤，在２００８年４月２日发给蔡旋的律师函中，迅雷指责超级兔子存在两大罪状。其中之一就是因为这款是对迅雷系列软件的非法修改，并通过超级兔子软件网站“非法向网络用户发布，获取非法利益”（不过迅雷并未就蔡旋所获个人利益进行举证）。迅雷据此要求超级兔子在２００８年４月１３日之前“停止修改及发行迅雷系列软件”。对于律师函蔡旋也做出了超快反应，对于超级兔子版的迅雷软件进行了删除。不过对于蔡旋的回应，迅雷仍然于４月底正式起诉了蔡旋，该案将在７月９日在广州正式开庭审理。迅雷状告蔡旋起诉书在起诉书中，我们看到迅雷起诉蔡旋的理由是严重侵犯原告的著作权：１、修改了原告与用户之间的软件许可协议；２、修改了软件的部分功能。对于所受的侵害，迅雷在诉讼中请求：１、请求法院判令被告立即停止侵犯原告著作权的行为；２、请求法院判令向原告公开赔礼道歉、消除影响；３、请求法院判令被告赔偿原告经济损失人民币伍拾万元；４、请求法院判令被告承担本案全部诉讼费用。迅雷状告蔡旋起诉书因为之前曾有过珊瑚虫版ＱＱ的案例，超级兔子版迅雷与之有着惊人的相似，所以对于本次案件的结果似乎已经比较清晰，除非蔡旋手中还有别的有利证据。“我也有证据，迅雷官方网站提供兔子版下载，看法官怎么判而已。”蔡旋认为自己跟“珊瑚虫”陈寿福的案例还有几点不同：１、珊瑚虫安装插件、替换广告，兔子没有这些。２、珊瑚虫提供的官方下载证据是第三方网站Ｃａｃｈｅ，并非直接采自ＱＱ网站。兔子证据是直接采自迅雷官方网站。３、迅雷已经发过律师信，兔子已经照做了，但仍然继续告上法院，说明迅雷律师信不讲信用！在珊瑚虫案尘埃落定之后，超级兔子案究竟向何处发展，之前法学界也曾有过许多争论，我们期待在现有的国家法律法规保护下，每个人都享受公正公平的公民权利。（责任编辑：韩建光）'
test_idlist_list = [content2idList(x)] 
test_X = kr.preprocessing.sequence.pad_sequences(test_idlist_list, seq_length)
id2label={0:'auto',1:'business', 2:'house',3:'it', 5:'travel', 6:'women', 4:'sports', 7:'yule'}
with tf.Session() as sess2:
    sess2.run(init)
    saver.restore(sess2,'H:/Anaconda/nlp_news/log/linermodel.cpkt')
    label=sess2.run(tf.argmax(predict_Y, 1),feed_dict={X_holder:test_X})
    print(sess2.run(embedding,feed_dict={X_holder:test_X}))
    print(sess2.run(tf.shape(embedding),feed_dict={X_holder:test_X}))
#print(id2label[int(label)])