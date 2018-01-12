# -*- coding: utf-8 -*-
import numpy as np
from numpy import mean,cov,cumsum,dot,linalg,size,flipud
from skimage import io
import sys

def readImage(path):
    image_ = io.imread(path)
    image_ = np.array(image_)
    image_ = image_.flatten()
    image_ = image_.reshape(1080000,1)
    return image_

def svd(ma_data):
    U, s, V = np.linalg.svd(ma_data, full_matrices=False)
    return U, s, V

def recons(ma_data, U, X_mean):
    eigen_face = U[:, :4]
    weight = np.dot(eigen_face.T,ma_data)
    recon = np.dot(eigen_face,weight) + X_mean
    recon -= np.min(recon, 0)
    recon /= np.max(recon, 0)
    M = (recon*255).astype(np.uint8)
    return M

readData = 1
if readData == 1:
    print('read image')
    image = readImage(sys.argv[1] + '/' + str(0)+'.jpg')
    for i in range(1,415):
        images = readImage(sys.argv[1] + '/' + str(i)+'.jpg')
        image = np.concatenate((image, images), axis=1)
    #np.save('data/image.npy',image)
    print('read image done')
else:
    image = np.load('data/image.npy')

X = image
X_mean = np.mean(X, axis=1)
X_mean = X_mean.reshape(1080000,1)
ma_data = X - X_mean
svdFlag = 1

if svdFlag == 1:
    print('SVD')
    U, s, V = svd(ma_data)
    print('SVD done')
    #np.save('data/U.npy',U)
    #np.save('data/s.npy',s)
    #np.save('data/V.npy',V)
else:
    U = np.load('data/U.npy')
    s = np.load('data/s.npy')
    V = np.load('data/V.npy')

M = recons(ma_data, U, X_mean)
print('reconstructing')

col = argv[2].split('.')[0]   
recoImageArray = M[:,col]
recoImage = np.reshape(recoImageArray, (600,600,3))
io.imsave('reconstruction.jpg',recoImage)
print('done')
'''======================Eigenfaces=============================='''
'''
for col in range(10):  
    eign_face = -U[:,col] 
    eign_face -= np.min(eign_face)
    eign_face /= np.max(eign_face)
    eign_face = (eign_face*255).astype(np.uint8) 
    eign_face = np.reshape(eign_face, (600,600,3))
    io.imsave('recon/'+str("eign_face_") + str(col) +'.jpg',eign_face)
'''
'''======================mean=============================='''
'''
X_mean -= np.min(X_mean, axis=0)
X_mean /= np.max(X_mean, axis=0)
X_mean = (X_mean*255).astype(np.uint8) 
X_mean = np.reshape(X_mean, (600,600,3))
io.imsave('recon/'+str("X_mean")+'.jpg',X_mean)
'''
'''
for col in range(5):    
    recoImageArray = recoImageMatrix[:,col]
    recoImage = np.reshape(recoImageArray, (600,600,3))
    io.imsave('recon/'+str(col)+'.jpg',recoImage)
'''