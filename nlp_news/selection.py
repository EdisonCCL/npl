# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 19:14:05 2018

@author: j
"""
import os
import random
root='H:\\Anaconda\\cnews\\news'
root1='H:\\Anaconda\\cnews\\news1'
for filename in os.listdir(root):
    with open(os.path.join(root,filename),'r',encoding='gb18030') as file:
        line_list = [k.strip() for k in file.readlines()]
        random.seed(filename)
        sample_list=random.sample(range(len(line_list)),6000)
        new_list=[line_list[k] for k in range(len(line_list)) if k in sample_list]
    with open(os.path.join(root1,filename),'w', encoding='gb18030') as file1:
        for i in new_list:
            file1.write(i+'\n')
'''
yule_list=all_list[0:6000]
auto_list=all_list[6000:12000]
sports_list=all_list[12000:18000]
women_list=all_list[18000:24000]
travel_list=all_list[24000:30000]
it_list=all_list[30000:36000]
business_list=all_list[36000:42000]
house_list=all_list[42000:48000]

with open('H:/Anaconda/cnews/news1/yule.txt','w', encoding='gb18030') as file:
    for i in yule_list:
        file.write(i+'\n')
with open('H:/Anaconda/cnews/news1/auto.txt','w', encoding='gb18030') as file:
    for i in auto_list:
        file.write(i+'\n')
with open('H:/Anaconda/cnews/news1/sports.txt', 'w',encoding='gb18030') as file:
    for i in sports_list:
        file.write(i+'\n')
with open('H:/Anaconda/cnews/news1/women.txt','w', encoding='gb18030') as file:
    for i in women_list:
        file.write(i+'\n')
with open('H:/Anaconda/cnews/news1/travel.txt', 'w',encoding='gb18030') as file:
    for i in travel_list:
        file.write(i+'\n')
with open('H:/Anaconda/cnews/news1/it.txt','w', encoding='gb18030') as file:
    for i in it_list:
        file.write(i+'\n')
with open('H:/Anaconda/cnews/news1/business.txt', 'w',encoding='gb18030') as file:
    for i in business_list:
        file.write(i+'\n')
with open('H:/Anaconda/cnews/news1/house.txt','w', encoding='gb18030') as file:
    for i in house_list:
        file.write(i+'\n')
'''