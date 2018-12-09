# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 22:30:14 2018

@author: j
"""

import os
from collections import Counter
root='H:\\Anaconda\\cnews\\news1'
root1='H:\\Anaconda\\cnews\\mynew'
new_list=[]
for filename in os.listdir(root):
    with open(os.path.join(root,filename),'r',encoding='gb18030') as file:
        line_list = [k.strip().split(maxsplit=1)[1] for k in file.readlines()]
#        train_content_list = [k.split(maxsplit=1)[1] for k in line_list]
    new_list=new_list+line_list

def getVocabularyList(content_list, vocabulary_size):
    allContent_str = ''.join(content_list)
    counter = Counter(allContent_str)
    vocabulary_list = [k[0] for k in counter.most_common(vocabulary_size)]
    return vocabulary_list

def makeVocabularyFile(content_list, vocabulary_size):
    vocabulary_list = getVocabularyList(content_list, vocabulary_size)
    with open(os.path.join(root1,'vocabulary.txt'), 'w', encoding='gb18030') as file:
        for vocabulary in vocabulary_list:
            file.write(vocabulary + '\n')

makeVocabularyFile(new_list, 5000)

