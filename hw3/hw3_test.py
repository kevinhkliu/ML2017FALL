# -*- coding: utf-8 -*-
import numpy as np
from keras.models import load_model
import csv
import sys

def loadData_test(path):
    count=0
    with open(path) as file:
        for line_id,line in enumerate(file):
            if count != 0:   
                label, X=line.split(',')              
                X = np.fromstring(X,dtype=int,sep=' ')
                X = X.reshape(-1, 48, 1)
                tempX = X[np.newaxis,:]
                if count == 1:    
                    X_ = tempX
                else:
                    X_ = np.concatenate((X_,tempX))
            count = count + 1
    return X_

def normalization(X):
    return X / 255

print("======load test data========")
x_test = loadData_test(sys.argv[1])
x_test = normalization(x_test)
print("======load test data Done========")

print("======load model ============")
model = load_model('CNN_model.hdf5')
print("======load model Done========")

prediction = model.predict_classes(x_test)
ans = []
for i in range(len(prediction)):
    ans.append([str(i)])
    ans[i].append(prediction[i])

filename = sys.argv[2]
text = open(filename, "w")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()    
print("-----DONE-----")
