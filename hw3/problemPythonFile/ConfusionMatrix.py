# -*- coding: utf-8 -*-
# -- coding: utf-8 --
from keras.models import load_model
from sklearn.metrics import confusion_matrix
import itertools
import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.jet):
    """
    This function prints and plots the confusion matrix.
    """
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def read_dataset(data_path):
    train_pixels = np.load(data_path)
    return train_pixels

def get_labels(data_path):
    train_labels = np.load(data_path)
    train_labels = train_labels.astype(int)
    return np.asarray(train_labels)


model = load_model('model/CNN_model.hdf5')
np.set_printoptions(precision=2)

dev_feats = read_dataset('data/X_val_WithoutOneHot.npy')

predictions = model.predict_classes(dev_feats)
#predictions = predictions.argmax(axis=-1)
print (predictions)

te_labels = get_labels('data/y_val_WithoutOneHot.npy')
print (len(te_labels))


conf_mat = confusion_matrix(te_labels,predictions)
print(np.mean(conf_mat))
plt.figure()
plot_confusion_matrix(conf_mat, classes=["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"])
plt.show()
