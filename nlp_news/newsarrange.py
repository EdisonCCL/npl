# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 09:22:55 2018

@author: j
"""

import os
import re
import jieba
#import chardet
#把每个文档的新闻内容和类别提取出来，分别作为列表的一个元素
root='E:\\谷歌浏览器下载内容\\sohunews'
x_train=[]
y_train=[]
'''for dirpath, dirnames, filenames in os.walk(root):
    for filepath in filenames:
'''
for filename in os.listdir(root):
    with open(os.path.join(root,filename),encoding='gb18030') as file:
        news=file.read()
        x=re.findall(r'<content>(.*)</content>',news)
        y=re.findall(r'<url>http://(\w*)[.]',news)
    x_train.append(x)
    y_train.append(y)
#把所有新闻内容和类别分别合并为一个列表
train_content=[]
train_label=[]
for i in x_train:
    train_content=train_content+i
for i in y_train:
    train_label=train_label+i


with open('H:/Anaconda/cnews/newtrain1.txt','w',encoding='gb18030') as file:
    for i,j in zip(train_label,train_content):
        file.write(i+' '+j+'\n')

#得到词库
'''
vocabs=[]
for i in train_content:
    vocab=jieba.lcut(i)
    vocabs=vocabs+vocab
vocabs=list(set(vocabs))
word2id_dict = dict([(b, a) for a, b in enumerate(vocabs)])
content2idList = lambda content : [word2id_dict[word] for word in content if word in word2id_dict]
train_idlist_list = [content2idList(content) for content in train_content]
'''   




