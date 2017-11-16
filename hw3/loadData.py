# -*- coding: utf-8 -*-
import numpy as np
from math import floor
import sys
def _shuffle(X, Y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])

def split_valid_set(X_all, Y_all, percentage):
    all_data_size = len(X_all)
    valid_data_size = int(floor(all_data_size * percentage))

    X_all, Y_all = _shuffle(X_all, Y_all)

    X_train, Y_train = X_all[0:valid_data_size], Y_all[0:valid_data_size]
    X_valid, Y_valid = X_all[valid_data_size:], Y_all[valid_data_size:]

    return X_train, Y_train, X_valid, Y_valid

def loadData_train(path):
    count=0
    with open(path) as file:
        for line_id,line in enumerate(file):
            if count != 0:   
                label, X=line.split(',')
                tempY = labelTransform(label) 
                
                X = np.fromstring(X,dtype=int,sep=' ')
                X = X.reshape(-1, 48, 1)
                tempX = X[np.newaxis,:]
                print(str(count) + "/28710")
                if count == 1:    
                    X_ = tempX
                    y = tempY
                else:
                    X_ = np.concatenate((X_,tempX))
                    y =  np.concatenate((y,tempY))
            count = count + 1
    return X_, y

def labelTransform(index):
    y = np.zeros(7)
    y = y.reshape(1,7)
    y[0][int(index)] = 1
    return y

def normalization(X):
    return X / 255

X, y = loadData_train(sys.argv[1])
X = normalization(X)

x_train, x_val, y_train, y_val = split_valid_set(X, y, 0.1)

# save model
np.save('data/x_train.npy', x_train)
np.save('data/y_train.npy', y_train)
np.save('data/X_val.npy', x_val)
np.save('data/y_val.npy', y_val)
print("------------------DONE-----------------")
