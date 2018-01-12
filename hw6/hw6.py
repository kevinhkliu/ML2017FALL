# -*- coding: utf-8 -*-
import numpy as np 
from sklearn.cluster import KMeans
import pandas as pd
import csv
from sklearn.decomposition import PCA
import sys

'''=======================Read Data==================='''
image = np.load(sys.argv[1])
image = image.astype('float32') / 255.

'''=======================PCA========================='''
pca = PCA(n_components=400, copy=True, whiten=True, svd_solver='full')
X_pca= pca.fit_transform(image)

'''=======================Read test===================='''
c = pd.read_csv(sys.argv[2])
image1_index = c['image1_index'].values
image2_index = c['image2_index'].values

'''======================Kmeans===================='''
kmeans = KMeans(n_clusters=2,random_state=0)
kmeans.fit(X_pca)
labels = kmeans.predict(X_pca)

'''====================testing====================='''
ans = []
for idx in range(len(image1_index)):
    ans.append([str(idx)])
    get_img1_index = image1_index[idx]
    get_img2_index = image2_index[idx]
    if labels[get_img1_index] == labels[get_img2_index]:
        ans[idx].append(1)
    else:
        ans[idx].append(0)
        
'''====================output====================='''      
filename = sys.argv[3]
text = open(filename, "w")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["ID","Ans"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()    
print("-----DONE-----")
