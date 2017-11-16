# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation
from keras.layers import Conv2D,MaxPooling2D,Flatten
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.client import device_lib
from keras.constraints import maxnorm#print(device_lib.list_local_devices())
x_train = np.load('data/x_train.npy')
y_train = np.load('data/y_train.npy')
x_val = np.load('data/X_val.npy')
y_val = np.load('data/y_val.npy')


model =Sequential()
'''-----------------cnn layer----------------'''
#cnn layer1
model.add(Conv2D(48,kernel_size = (3, 3),input_shape=(48,48,1),activation='relu'))
model.add(Conv2D(48, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
#cnn layer2 
model.add(Conv2D(96, (3, 3), activation='relu'))
model.add(Conv2D(96, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


'''-----------------Fully Connected Feedward network----------------'''
# flatten and fully connect
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
'''--------------------------------------------------------------'''
datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.15
        ) 

datagen.fit(x_train)
trainData = datagen.flow(x_train, y_train, batch_size=512)

# checkpoint
filepath="model/CNN_model.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=2, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

#model.fit(x_train,y,validation_split=0.2,batch_size=400,epochs=300, callbacks=callbacks_list,verbose=0)
history = model.fit_generator(trainData,steps_per_epoch=1500,epochs=90,verbose=1,callbacks=callbacks_list,validation_data=(x_val,y_val))

result = model.evaluate(x_train, y_train)
print("\nTrain Acc: ", result[1])

model.summary()
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Accuracy plot of CNN model')
plt.ylabel('Accuracy')
plt.xlabel('# of epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()
