# -*- coding: utf-8 -*-
import numpy as np
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, add, dot, Flatten
from keras.callbacks import ModelCheckpoint
#import matplotlib.pyplot as plt

user_train = np.load('data/user_train.npy')
user_val = np.load('data/user_val.npy') 
movie_train = np.load('data/movie_train.npy') 
movie_val = np.load('data/movie_val.npy') 
rating_train = np.load('data/rating_train.npy') 
rating_val = np.load('data/rating_val.npy') 

n_users = 6040
n_movies = 3952
lantent_factor = 512

user_input = Input(shape=(1,), dtype='int64', name='user_input')
user_embedding = Embedding(n_users, lantent_factor)(user_input)

movie_input = Input(shape=(1,), dtype='int64', name='movie_input')
movie_embedding = Embedding(n_movies, lantent_factor)(movie_input)

user_bias = Embedding(input_dim=n_users, output_dim=1, name='user_bias', input_length=1)(user_input)
movie_bias = Embedding(input_dim=n_movies, output_dim=1, name='movie_bias', input_length=1)(movie_input)

predicted_preference = dot(inputs=[user_embedding, movie_embedding], axes=2)
predicted_preference = Flatten()(predicted_preference)

predicted_preference = add(inputs=[predicted_preference, movie_bias, user_bias])
predicted_preference = Flatten()(predicted_preference)

model = Model(inputs=[user_input,movie_input], outputs=predicted_preference)
model.compile(loss='mse', optimizer='adamax')

filepath="model/model.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

history = model.fit([user_train, movie_train], rating_train,
          batch_size=40000,
          epochs=60,
          callbacks=callbacks_list,
          validation_data=([user_val, movie_val], rating_val))

'''    
model.summary()

# summarize history for accuracy
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('loss plot of model(withoutNormalization)')
plt.ylabel('Loss')
plt.xlabel('# of epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()
'''