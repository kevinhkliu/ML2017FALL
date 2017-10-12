# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 01:23:24 2017

@author: kevin
"""
import csv 
import numpy as np
import math

data = []
# 每一個維度儲存一種污染物的資訊
for i in range(18):
	data.append([])

n_row = 0
text = open('data/train.csv', 'r', encoding='big5') 
row = csv.reader(text , delimiter=",")
for r in row:
    # 第0列沒有資訊
    if n_row != 0:
        # 每一列只有第3-27格有值(1天內24小時的數值)
        for i in range(3,27):
            if r[i] != "NR":
                data[(n_row-1)%18].append(float(r[i]))
            else:
                data[(n_row-1)%18].append(float(0))	
    n_row = n_row + 1
text.close()

numdays = 20
monthPeriods = numdays * 24
segment = 5
timeSeries = monthPeriods - segment

x = []
y = []
# 每 12 個月
for i in range(12):
    # 一個月取連續10小時的data可以有471筆
    for j in range(timeSeries):
        x.append([])
        # 18種污染物
        for t in range(18):
            # 連續9小時
            for s in range(segment):
                x[timeSeries*i+j].append(data[t][monthPeriods*i+j+s])
        y.append(data[9][monthPeriods*i+j+segment])
text.close()
x = np.array(x)
y = np.array(y)

print(x)

# add square term
x = np.concatenate((x,x**2), axis=1)

x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)
w = np.zeros(len(x[0]))
l_rate = 10
repeat = 1300000
λ = 0.001
x_t = x.transpose()
s_gra = np.zeros(len(x[0]))

for i in range(repeat):
    loss = np.dot(x,w) - y
    RMSE  = math.sqrt(np.sum(loss**2) / len(x))
    temp1_w = w[1:]
    temp_w = np.concatenate((np.zeros(1),temp1_w),axis=0)
    gra = np.dot(x_t,loss) + 2 * λ * temp_w 
   
    s_gra += gra**2
    ada = np.sqrt(s_gra)
    w = w - l_rate * gra/ada
    print ('iteration: %d | Cost: %f  ' % ( i,RMSE))
    
    
# save model
np.save('all_model.npy',w)
