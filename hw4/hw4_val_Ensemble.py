# -*- coding: utf-8 -*-
import numpy as np
import csv
from keras.models import load_model
import sys

print("======load test data========")
x_test = np.load('data/x_test.npy')
print("======load test data Done========")

print("======load model ============")
modelGRU1 = load_model('GRU_model_nolabel0.hdf5')
modelGRU2 = load_model('GRU_model_nolabel1.hdf5')
modelGRU3 = load_model('GRU_model_nolabel2.hdf5')
print("======load model Done========")

'''
print("======load val data========")
x_val = np.load('data/x_val.npy')
y_val = np.load('data/y_val.npy')
print("======load val data Done========")

predictionGRU1 = modelGRU1.predict(x_val, batch_size=1000)
predictionGRU2 = modelGRU2.predict(x_val, batch_size=1000)
predictionGRU3 = modelGRU3.predict(x_val, batch_size=1000)

correct = 0.0
for i in range(len(predictionGRU1)):
    if predictionGRU1[i][0] > 0.5:
        predict_label = '1'
    else:
        predict_label = '0'
    
    if y_val[i] == predict_label:
        correct = correct + 1
GRU1accurracy = float(correct) / len(predictionGRU1)

correct = 0.0
for i in range(len(predictionGRU2)):
    if predictionGRU2[i][0] > 0.5:
        predict_label = '1'
    else:
        predict_label = '0'
    
    if y_val[i] == predict_label:
        correct = correct + 1
GRU2accurracy = float(correct) / len(predictionGRU2)

correct = 0.0
for i in range(len(predictionGRU3)):
    if predictionGRU3[i][0] > 0.5:
        predict_label = '1'
    else:
        predict_label = '0'
    if y_val[i] == predict_label:
        correct = correct + 1
GRU3accurracy = float(correct) / len(predictionGRU3)

correct = 0.0
for i in range(len(predictionGRU1)):
    if (predictionGRU1[i][0] + predictionGRU2[i][0]+ predictionGRU3[i][0]) / 3 > 0.5:
        predict_label = '1'
    else:
        predict_label = '0'
    if y_val[i] == predict_label:
        correct = correct + 1
accurracy = float(correct) / len(predictionGRU1)

print("GRU1 accuracy: ", GRU1accurracy)
print("GRU2 accuracy: ", GRU2accurracy)
print("GRU3 accuracy: ", GRU3accurracy)
print("ensemble accuracy: ", accurracy)

'''
print("====== TESTING =======")
predictionGRU1 = modelGRU1.predict(x_test, batch_size=1000)
predictionGRU2 = modelGRU2.predict(x_test, batch_size=1000)
predictionGRU3 = modelGRU3.predict(x_test, batch_size=1000)

ans = []
for i in range(len(predictionGRU1)):
    finalProb = (predictionGRU1[i][0] + predictionGRU2[i][0]+ predictionGRU3[i][0]) / 3 
    ans.append([str(i)])
    if finalProb <=0.5:
        ans[i].append(0)
    else:
        ans[i].append(1)

filename = sys.argv[1]
text = open(filename, "w")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()    
print("-----DONE-----")
