# -*- coding: utf-8 -*-
import numpy as np
from numpy import mean,cov,cumsum,dot,linalg,size,flipud
from skimage import io
import sys, os


#read recon image
test = os.path.join(sys.argv[1],sys.argv[2])
reco = io.imread(test).reshape(-1)[:,np.newaxis].astype('float32')

#read image
print('read image ')    
dirs = os.listdir(sys.argv[1] + '/')
image = []
for file in dirs:
    pic = io.imread(sys.argv[1] + file).flatten()
    image.append(pic)
print('read image done')
train = np.array(image).T
mean = np.mean(train,axis=1)[:,np.newaxis]
ma_data = train - mean
reco -= mean


#SVD
U,s,v = np.linalg.svd(ma_data,full_matrices=False)

#reconstructing
eigen_face = U[:,0:4]
test = np.dot(eigen_face.T,reco)
recon = np.dot(eigen_face,test) + mean
recon -= np.min(recon,0)
recon /= np.max(recon,0)
M = (recon*255).astype(np.uint8)
reco = M.reshape(600,600,3)

#image save 
io.imsave('reconstruction.jpg',reco)
