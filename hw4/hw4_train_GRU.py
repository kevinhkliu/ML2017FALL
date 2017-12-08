# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
from keras.callbacks import ModelCheckpoint
import sys

x_train = np.load('data/x_train.npy')
y_train = np.load('data/y_train.npy')
x_val = np.load('data/x_val.npy')
y_val = np.load('data/y_val.npy')
embedding_matrix = np.load('data/gensim_word2vec.npy') 

def train_GRU(x_train,y_train,x_val,y_val,embedding_matrix):
    max_features = 80000
    max_length = 30
    embedding_size = 300
    gru_output_size = 128
    batch_size = 1000
    epochs = 15
      
    model = Sequential()
    model.add(Embedding(input_dim=max_features, output_dim=embedding_size, 
                        weights=[embedding_matrix], input_length=max_length, trainable=False))
    model.add(Dropout(0.25))
    
    model.add(GRU(embedding_size, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
    model.add(GRU(gru_output_size, dropout=0.2, recurrent_dropout=0.2))

    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(units=1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])
    
    filepath="model/GRU_model.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=2, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=callbacks_list)
    
    scores = model.evaluate(x_train, y_train, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))

    model.summary()
    #summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Accuracy plot of GRU model')
    plt.ylabel('Accuracy')
    plt.xlabel('# of epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()

    return model

def predict_nolabel(modelGRU):
    twitter_List_GRU=[]
    label_List_GRU=[]
    x_train_nolabel = np.load('data/x_train_nolabel.npy')
    prediction_GRU = modelGRU.predict(x_train_nolabel, batch_size=1000)
    for i in range(len(prediction_GRU)):
        if prediction_GRU[i][0] >= 0.95:
            twitter_List_GRU.append(x_train_nolabel[i])
            label_List_GRU.append(1)
        elif prediction_GRU[i][0] <= 0.05:
            twitter_List_GRU.append(x_train_nolabel[i])
            label_List_GRU.append(0)
    return twitter_List_GRU, label_List_GRU


def train_GRU_nolabel(train,y_train,x_val,y_val,embedding_matrix,twitter_List_GRU,label_List_GRU):
    max_features = 80000
    max_length = 30
    embedding_size = 300
    gru_output_size = 128
    batch_size = 1000
    epochs = 15
    
    total_X_train = np.concatenate((x_train, twitter_List_GRU), axis=0) 
    total_y_train = np.concatenate((y_train, label_List_GRU), axis=0) 

    model = Sequential()
    model.add(Embedding(input_dim=max_features, output_dim=embedding_size, 
                        weights=[embedding_matrix], input_length=max_length, trainable=False))
    model.add(Dropout(0.25))
    
    model.add(GRU(embedding_size, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
    model.add(GRU(gru_output_size, dropout=0.2, recurrent_dropout=0.2))
    

    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(units=1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])
    
    
    filepath="model/GRU_model_nolabel.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=2, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    
    history = model.fit(total_X_train, total_y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=callbacks_list)
    
    scores = model.evaluate(x_train, y_train, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    
    model.summary()
    #summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Accuracy plot of GRU model (withNolabel)')
    plt.ylabel('Accuracy')
    plt.xlabel('# of epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()
    
modelGRU = train_GRU(x_train,y_train,x_val,y_val,embedding_matrix)
twitter_List_GRU, label_List_GRU = predict_nolabel(modelGRU)
train_GRU_nolabel(x_train,y_train,x_val,y_val,embedding_matrix,twitter_List_GRU,label_List_GRU)