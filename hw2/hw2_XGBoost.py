# -*- coding: utf-8 -*-
import sys
import csv 
import numpy as np
from xgboost import XGBClassifier
#from sklearn.model_selection import RandomizedSearchCV
#from time import time
#from scipy.stats import randint as sp_randint
#from scipy.stats import uniform as sp_uniform
'''
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
            
def RParameter_Search(clf,param_dist,iteration,cv,X,y):
    # run randomized search
    n_iter_search = iteration
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist,cv=cv,
                                   n_iter=n_iter_search)
    start = time()
    random_search.fit(X, y)
    #print (random_search)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
    report(random_search.cv_results_)
'''   
X=[]
y=[]
X_test=[]
'''-------X_train--------------'''
text = open(sys.argv[3], 'r', encoding='big5')
row = csv.reader(text , delimiter=",")
for r in row:
    X.append(r)
text.close()
X = np.array(X)
X = np.delete(X, (0), axis=0)
X = X.astype(float)
'''-------Y_train-----------------'''
text = open(sys.argv[4], 'r', encoding='big5')
row = csv.reader(text , delimiter=",")
for r in row:
    y.append(r)
text.close()
y = np.array(y)
y  = y.reshape((len(y),)) 
y = np.delete(y, (0), axis=0)
y = y.astype(float)
'''-------X_test-----------------'''
text = open(sys.argv[5], 'r', encoding='big5')
row = csv.reader(text , delimiter=",")
for r in row:
    X_test.append(r)
text.close()
X_test = np.array(X_test)
X_test = np.delete(X_test, (0), axis=0)
X_test = X_test.astype(float)
'''-------searchBestParameters and cross validation-----------------'''
'''
model = XGBClassifier()
XGB_param_dist = {
             "n_estimators":sp_randint(100,1000),         
              "max_depth": sp_randint(6, 12),
              "learning_rate" :  sp_uniform(0.001, 10),
                  }
RParameter_Search(model,XGB_param_dist,50,4,X,y)
'''
'''-------XGBoost-----------------'''
model = XGBClassifier(learning_rate=0.01824058894243874, max_depth= 6, n_estimators=992)
model.fit(X, y)
print(model.score(X, y)) 

# generate class probabilities
predicted = model.predict(X_test)

ans = []
for i in range(len(X_test)):
    ans.append([str(i+1)])
    ans[i].append(int(predicted[i]))
    
filename = sys.argv[6]
text = open(filename, "w")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()
print("-------------Done---------------") 