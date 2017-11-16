# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 14:09:27 2017

@author: USER
"""
from keras.utils.vis_utils import plot_model
from keras.models import load_model


model = load_model('model/CNN_model.hdf5')
plot_model(model, to_file='result/model.png')