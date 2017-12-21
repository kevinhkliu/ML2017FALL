# -*- coding: utf-8 -*-
import csv
from keras.models import load_model
import sys
import numpy as np


def loadData_train(path):
    user_list = []
    movie_list = []
    with open(path) as file:
        for line_id,line in enumerate(file):
            if line_id != 0:   
                label, UserID, MovieID=line.split(',')
                user_list.append(UserID)
                movie_list.append(MovieID)     
    return user_list, movie_list
user_list, movie_list = loadData_train(sys.argv[1])
user_list = np.asarray(user_list)
movie_list = np.asarray(movie_list)
print("======load model ============")
model1 = load_model('model1.hdf5')
model2 = load_model('model2.hdf5')
model3 = load_model('model3.hdf5')
print("======load model Done========")
'''
print("======load val data========")
user_val = np.load('data/user_val.npy') 
movie_val = np.load('data/movie_val.npy') 
rating_val = np.load('data/rating_val.npy') 
print("======load val data Done========")

predictionModel1 = model1.predict([user_val,movie_val], batch_size=1000)
predictionModel2 = model2.predict([user_val,movie_val], batch_size=1000)
predictionModel3 = model3.predict([user_val,movie_val], batch_size=1000)

error = 0.0
for i in range(len(predictionModel1)): 
    temp = predictionModel1[i][0] - rating_val[i]
    error = error + temp * temp

model1_loss = float(error) / len(predictionModel1)


error = 0.0
for i in range(len(predictionModel2)): 
    temp = predictionModel2[i][0] - rating_val[i]
    error = error + temp * temp

model2_loss = float(error) / len(predictionModel2)

error = 0.0
for i in range(len(predictionModel3)): 
    temp = predictionModel3[i][0] - rating_val[i]
    error = error + temp * temp

model3_loss = float(error) / len(predictionModel3)


error = 0.0
for i in range(len(predictionModel1)): 
    avg_rating = (predictionModel1[i][0] + predictionModel2[i][0] + predictionModel3[i][0])/3
    temp = avg_rating - rating_val[i]
    error = error + temp * temp

loss = float(error) / len(predictionModel1)

print("model1 loss: ", model1_loss)
print("model2 loss: ", model2_loss)
print("model3 loss: ", model3_loss)
print("ensemble loss: ", loss)

'''
print("====== TESTING =======")
predictionModel1 = model1.predict([user_list,movie_list], batch_size=1000)
predictionModel2 = model2.predict([user_list,movie_list], batch_size=1000)
predictionModel3 = model3.predict([user_list,movie_list], batch_size=1000)

ans = []
for i in range(len(predictionModel1)):
    final_rating = (predictionModel1[i][0] + predictionModel2[i][0]+ predictionModel3[i][0]) / 3 
    ans.append([str(i+1)])
    ans[i].append(final_rating)

filename = sys.argv[2]
text = open(filename, "w")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["TestDataID","Rating"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()    
print("-----DONE-----")
