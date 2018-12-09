# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 10:39:54 2018

@author: j
"""
import os
import random
with open('H:/Anaconda/cnews/newtrain1.txt', encoding='gb18030') as file:
    line_list = [k.strip() for k in file.readlines() if len(k)>100]
    train_label_list = [k.split()[0] for k in line_list]
    train_content_list = [k.split(maxsplit=1)[1] for k in line_list]
'''
i=0
while i <len(train_label_list):
    if train_label_list[i]=='mil' or train_label_list[i]=='2008' or train_label_list[i]=='career' or train_label_list[i]=='learning' or train_label_list[i]=='cul' or train_label_list[i]=='health':
        train_label_list.pop(i)
        train_content_list.pop(i)
        continue
    i=i+1
'''
yule_list=[]
auto_list=[]
sports_list=[]
women_list=[]
travel_list=[]
it_list=[]
business_list=[]
house_list=[]

for i in range(len(train_label_list)):
    if train_label_list[i]=='yule':
        yule_list.append(train_content_list[i])
    elif train_label_list[i]=='auto':
        auto_list.append(train_content_list[i])
    elif train_label_list[i]=='sports':
        sports_list.append(train_content_list[i])
    elif train_label_list[i]=='women':
        women_list.append(train_content_list[i])
    elif train_label_list[i]=='travel':
        travel_list.append(train_content_list[i])
    elif train_label_list[i]=='it':
        it_list.append(train_content_list[i])
    elif train_label_list[i]=='business':
        business_list.append(train_content_list[i])
    elif train_label_list[i]=='house':
        house_list.append(train_content_list[i])
with open('H:/Anaconda/cnews/news/yule.txt','w', encoding='gb18030') as file:
    for i in yule_list:
        file.write('yule '+i+'\n')
with open('H:/Anaconda/cnews/news/auto.txt','w', encoding='gb18030') as file:
    for i in auto_list:
        file.write('auto '+i+'\n')
with open('H:/Anaconda/cnews/news/sports.txt', 'w',encoding='gb18030') as file:
    for i in sports_list:
        file.write('sports '+i+'\n')
with open('H:/Anaconda/cnews/news/women.txt','w', encoding='gb18030') as file:
    for i in women_list:
        file.write('women '+i+'\n')
with open('H:/Anaconda/cnews/news/travel.txt', 'w',encoding='gb18030') as file:
    for i in travel_list:
        file.write('travel '+i+'\n')
with open('H:/Anaconda/cnews/news/it.txt','w', encoding='gb18030') as file:
    for i in it_list:
        file.write('it '+i+'\n')
with open('H:/Anaconda/cnews/news/business.txt', 'w',encoding='gb18030') as file:
    for i in business_list:
        file.write('business '+i+'\n')
with open('H:/Anaconda/cnews/news/house.txt','w', encoding='gb18030') as file:
    for i in house_list:
        file.write('house '+i+'\n')

'''
for filename in os.listdir(root):
    with open(os.path.join(root,filename),'r',encoding='gbk',errors='ignore') as file:
        line_list = [k.strip() for k in file.readlines()]
        sample_list=random.sample(range(len(line_list)),6000)
        new_list=[line_list[k] for k in range(len(line_list)) if k in sample_list]
'''


'''x1=random.sample(range(len(yule_list)),6000)
x2=random.sample(range(len(auto_list)),6000)
x3=random.sample(range(len(sports_list)),6000)
x4=random.sample(range(len(women_list)),6000)
x5=random.sample(range(len(travel_list)),6000)
x6=random.sample(range(len(it_list)),6000)
x7=random.sample(range(len(business_list)),6000)
x8=random.sample(range(len(house_list)),6000)

yule_lists=[yule_list[k] for k in range(len(yule_list)) if k in x1]
yule_lists=[yule_list[k] for k in range(len(yule_list)) if k in x1]
yule_lists=[yule_list[k] for k in range(len(yule_list)) if k in x1]
yule_lists=[yule_list[k] for k in range(len(yule_list)) if k in x1]    
yule_lists=[yule_list[k] for k in range(len(yule_list)) if k in x1]    
yule_lists=[yule_list[k] for k in range(len(yule_list)) if k in x1]       
yule_lists=[yule_list[k] for k in range(len(yule_list)) if k in x1]       
yule_lists=[yule_list[k] for k in range(len(yule_list)) if k in x1]
 '''       
'''
from collections import Counter

def getVocabularyList(content_list, vocabulary_size):
    allContent_str = ''.join(content_list)
    counter = Counter(allContent_str)
    vocabulary_list = [k[0] for k in counter.most_common(vocabulary_size)]
    return vocabulary_list

def makeVocabularyFile(content_list, vocabulary_size):
    vocabulary_list = getVocabularyList(content_list, vocabulary_size)
    with open('vocabulary.txt', 'w', encoding='utf8') as file:
        for vocabulary in vocabulary_list:
            file.write(vocabulary + '\n')

makeVocabularyFile(train_content_list, 5000)
'''