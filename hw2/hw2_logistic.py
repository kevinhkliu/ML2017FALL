import pandas as pd
import csv
import numpy as np
import math
from math import log, floor
from random import seed
from random import randrange
import sys

def load_data(train_data_path, train_label_path, test_data_path):
    X_train = pd.read_csv(train_data_path, sep=',', header=0)
    X_train = np.array(X_train.values)
    Y_train = pd.read_csv(train_label_path, sep=',', header=0)
    Y_train = np.array(Y_train.values)
    X_test = pd.read_csv(test_data_path, sep=',', header=0)
    X_test = np.array(X_test.values)
    return (X_train, Y_train, X_test)

def normalize(X_all, X_test):
    # Feature normalization with train and test X
    X_train_test = np.concatenate((X_all, X_test))
    mu = (sum(X_train_test) / X_train_test.shape[0])
    sigma = np.std(X_train_test, axis=0)
    mu = np.tile(mu, (X_train_test.shape[0], 1))
    sigma = np.tile(sigma, (X_train_test.shape[0], 1))
    X_train_test_normed = (X_train_test - mu) / sigma

    # Split to train, test again
    X_all = X_train_test_normed[0:X_all.shape[0]]
    X_test = X_train_test_normed[X_all.shape[0]:]
    return X_all, X_test

def sigmoid(z):
    res = 1 / (1.0 + np.exp(-z))
    return np.clip(res, 1e-8, 1-(1e-8))


def cost(w, X, y):
    p_1 = sigmoid(np.dot(X, w)) # predicted probability of label 1
    log_l = (-y)*np.log(p_1) - (1-y)*np.log(1-p_1) # log-likelihood vector
    return log_l.mean()

def grad(w, X, y):
    p_1 = sigmoid(np.dot(X, w))
    X_t = X.transpose()
    error = p_1 - y # difference between label and prediction
    #grad = np.dot(error, p_1) / y.size # gradient vector
    grad = np.dot(X_t, error) / y.size # gradient vector
    return grad

# Learn coefficients using gradient descent
def logistic_regression(X, y, l_rate, iterations, λ, learn_M):
    X = np.concatenate((np.ones((X.shape[0],1)),X), axis=1)#add bias
    w = np.zeros(len(X[0]))
    X_t = X.transpose()
    s_gra = np.zeros(len(X[0]))
    y = y.reshape((len(X),))

    for i in range(iterations):
        if (i % 100) == 0:
            print ('iteration: %d ' % i) 
        F = sigmoid(np.dot(X,w))
        loss = F - y
        temp1_w = w[1:]
        temp_w = np.concatenate((np.zeros(1),temp1_w),axis=0)
        gra = (np.dot(X_t,loss) + λ * temp_w)/y.size 
        #print(np.dot(X_t,loss))
        s_gra += gra**2
        ada = np.sqrt(s_gra)
        if learn_M == 1:
            # avoid divide by zero
            for j in range(w.size):
                if ada[j] != 0:
                    w[j] = w[j] - l_rate * gra[j]/ada[j]
        else:
            if learn_M == 2:
                    w = w - l_rate * gra/ada
            else:
                w = w - (l_rate * gra)
    return w
    
def logistic_func(wX):
    return 1 / (1 + np.exp(-wX))
 
def cal_accuracy(y_actual, y_predicted):
	correct = 0
	for i in range(len(y_actual)):
		if y_actual[i] == y_predicted[i]:
			correct += 1
	return correct / float(len(y_actual)) * 100.0 

#def logistic_func(w, X):
#    return float(1) / (1 + np.exp(-np.dot(X,w)))

def predict(w, X_test):
    predicted_values = []
    X_test = np.concatenate((np.ones((X_test.shape[0],1)),X_test), axis=1)
    for i in range(len(X_test)):
        #pred_prob = sigmoid(np.dot(X_test[i],w))
        wX = np.dot(X_test[i], w)
        pred_prob = sigmoid(wX)
        if pred_prob >= 0.5:
            predicted_values.append(1)
        else :
            predicted_values.append(0)
    return predicted_values

# Split a dataset into k folds
def cross_split(X_data, y_data, kfolds):
    X_dataFolds = list()
    X_dataSet = list(X_data)
    y_dataFolds = list()
    y_dataSet = list(y_data)
    fold_size = int(len(X_data) / kfolds)
    for i in range(kfolds-1):
        X_fold = list()
        y_fold = list()
        while len(X_fold) < fold_size:
            index = randrange(len(X_dataSet))
            X_fold.append(X_dataSet.pop(index))
            y_fold.append(y_dataSet.pop(index))

        X_dataFolds.append(X_fold)
        y_dataFolds.append(y_fold)
 
    X_dataFolds.append(X_dataSet)
    y_dataFolds.append(y_dataSet)
    return X_dataFolds, y_dataFolds

# Evaluate an algorithm using a cross validation split
def eval_algorithm(X_data, y_data, X_testData, kfolds, n_flag, l_rate, iterations, λ, learn_M):
    if n_flag == 1:
        X_data, X_testData = normalize(X_data, X_testData)
    X_dataFolds, y_dataFolds = cross_split(X_data, y_data, kfolds)
    scores = list()
    theta = list()
    
    for i in range(len(X_dataFolds)):
        print("fold", i)
        #generate X_train and X_test
        X_train = list()
        for j in range(len(X_dataFolds)):
            if i != j:
                X_train.append(X_dataFolds[j])
        if (len(X_dataFolds) == 1):
            X_train.append(X_dataFolds[i])
        X_train = sum(X_train, [])
        X_test = list(X_dataFolds[i])
        X_train = np.array(X_train)
        X_test = np.array(X_test)
 
         #generate y_train and y_test
        y_train = list()
        for j in range(len(y_dataFolds)):
            if i != j:
                y_train.append(y_dataFolds[j])
        if (len(y_dataFolds) == 1):
            y_train.append(y_dataFolds[i])
        y_train = sum(y_train, [])
        y_test = list(y_dataFolds[i])
        y_train = np.array(y_train)
        y_test = np.array(y_test)

        w = logistic_regression(X_train, y_train, l_rate, iterations, λ, learn_M)
        y_predicted = predict(w, X_test)
        accuracy = cal_accuracy(y_test, y_predicted)
        theta.append(w)
        scores.append(accuracy)
    return theta, scores , X_testData  


X, y, X_test = load_data(sys.argv[3], sys.argv[4], sys.argv[5])
'''-------training-----------------'''
seed(1)
l_rate = 1  # 3 best for non_normalize; 10 is not good
iterations = 2000
λ =0.0001
kfolds = 5
learn_M = 1  # adjust theta (w), 1, 2, 3
n_flag = 1  # 1 : normalization
theta, scores, X_testData = eval_algorithm(X, y, X_test, kfolds, n_flag, l_rate, iterations, λ, learn_M)

maxscore = 0;
maxi = 0;
for i in range(kfolds):
    print(theta[i])
    if scores[i] > maxscore:
        maxscore = scores[i]
        maxi = i
print("i maxscore:", i, ":", scores[maxi])       
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))   
# save model
np.save('model/logistic_model.npy',theta[maxi])



'''-------predict-----------------'''
'''-------read model-----------------'''
w = np.load('model/logistic_model.npy')
'''-------X_test-----------------'''
x_test = np.concatenate((np.ones((X_testData.shape[0],1)),X_testData), axis=1)

ans = []
for i in range(len(x_test)):
    ans.append([str(i+1)])
    #a = np.dot(x_test[i],w)
    a = float(1) / (1 + math.e**(-x_test[i].dot(w)))
    if a >= 0.5:
       ans[i].append(1)
    else :
       ans[i].append(0)
      
filename = sys.argv[6]
text = open(filename, "w")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()
print("-------------Done---------------") 

