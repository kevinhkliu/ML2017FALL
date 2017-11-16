# -*- coding: utf-8 -*-
from keras.models import load_model
import matplotlib.pyplot as plt
from vis.visualization import visualize_saliency
from vis.utils import utils
from keras import activations
import numpy as np

def read_dataset(data_path):
    train_pixels = np.load(data_path)
    return train_pixels

idx = 19
private_pixels = read_dataset('data/X_val_WithoutOneHot.npy')
img = private_pixels[idx].reshape(1,48,48,1)

#print original picture
plt.figure()
plt.imshow(private_pixels[idx].reshape(48,48), cmap='gray')
plt.colorbar()
plt.tight_layout()
fig = plt.gcf()
plt.draw()
fig.savefig("result/original.png", dpi=100)

#load model
model = load_model('model/CNN_model.hdf5')
model.layers[-1].activation = activations.linear
model = utils.apply_modifications(model)

#pred & call function
pred = model.predict_classes(img)
heatmap = visualize_saliency(model, layer_idx=-1, filter_indices=pred, seed_input=img)

#print heatMap
plt.figure()
plt.imshow(heatmap,cmap='jet')
plt.colorbar()
plt.tight_layout()
fig = plt.gcf()
plt.draw()
fig.savefig("result/cmap.png", dpi=100)

#Mask
heatmap = heatmap[:,:,2].astype('float32')
heatmap = heatmap/255
thres = 0.75
see = private_pixels[idx].reshape(48,48)
see[np.where(heatmap <= thres)] = np.mean(see)

#print Mask
plt.figure()
plt.imshow(see,cmap='gray')
plt.colorbar()
plt.tight_layout()
fig = plt.gcf()
plt.draw()
fig.savefig("result/gray.png", dpi=100)
