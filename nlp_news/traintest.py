# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 22:45:59 2018

@author: j
"""

import os
root='H:\\Anaconda\\cnews\\news1'
root1='H:\\Anaconda\\cnews\\mynew'
new_list=[]
for filename in os.listdir(root):
    with open(os.path.join(root,filename),'r',encoding='gb18030') as file:
        line_list = [k.strip() for k in file.readlines()]
    new_list=new_list+line_list

train_list=new_list[0:5000]+new_list[6000:11000]+new_list[12000:17000]+new_list[18000:23000]+new_list[24000:29000]+new_list[30000:35000]+new_list[36000:41000]+new_list[42000:47000]
test_list=new_list[5000:6000]+new_list[11000:12000]+new_list[17000:18000]+new_list[23000:24000]+new_list[29000:30000]+new_list[35000:36000]+new_list[41000:42000]+new_list[47000:48000]

with open(os.path.join(root1,'train.txt'), 'w', encoding='gb18030') as file:
    for i in train_list:
        file.write(i+ '\n')
with open(os.path.join(root1,'test.txt'), 'w', encoding='gb18030') as file:
    for i in test_list:
        file.write(i+ '\n')