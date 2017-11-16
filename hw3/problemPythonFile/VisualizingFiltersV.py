from keras.models import load_model
import matplotlib.pyplot as plt
from keras import backend as K
#from termcolor import colored,cprint
import numpy as np
import time
nb_class = 7
LR_RATE = 0.02
NUM_STEPS = 100
#RECORD_FREQ = 10
NUM_ITERS = 20
def read_dataset(data_path):
    train_pixels = np.load(data_path)
    return train_pixels

def get_labels(data_path):
    train_labels = np.load(data_path)
    train_labels = train_labels.astype(int)
    return np.asarray(train_labels)
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    #x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x
def normalize(x):
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def grad_ascent(num_step,input_image_data,iter_func):
    for i in range(0,num_step):
        loss, grads = iter_func([input_image_data,1])
        #grads = normalize(K.gradients(loss,input_image_data)[0])
        input_image_data =  input_image_data + grads * LR_RATE
        input_image_data = input_image_data * 0.9
    filter_images = input_image_data 

    return filter_images, loss


emotion_classifier = load_model('model/CNN_model.hdf5')
layer_dict = dict([layer.name, layer] for layer in emotion_classifier.layers[1:])
input_img = emotion_classifier.input

# visualize the area CNN see
mode = 2
if mode == 1:
    collect_layers = list()
    collect_layers.append(K.function([input_img,K.learning_phase()],[layer_dict['conv2d_22'].output]))

    dev_feat = read_dataset('data/X_val_WithoutOneHot.npy')
    dev_label = get_labels('data/y_val_WithoutOneHot.npy')
    choose_id = 19
    photo = dev_feat[choose_id]
   

    for cnt, fn in enumerate(collect_layers):
        im = fn([photo.reshape(1,48,48,1),0])
        fig = plt.figure(figsize=(14,8))
        nb_filter = im[0].shape[3]
        for i in range(nb_filter):
            ax = fig.add_subplot(nb_filter/16,16,i+1)
            ax.imshow(im[0][0,:,:,i],cmap='Blues')
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
            plt.tight_layout()
        fig.suptitle('Output of layer{} (Given image{})'.format(cnt,choose_id))
        fig.savefig("result/Visualizing_Filters1", dpi=100)

else:
    name_ls = ['conv2d_22']
    collect_layers = list()
    collect_layers.append(layer_dict[name_ls[0]].output)
    
    #dev_feat = read_dataset('data/X_val_WithoutOneHot.npy')
    #choose_id = 189
    #photo = dev_feat[choose_id]
    
    for cnt, c in enumerate(collect_layers):
        filter_imgs = [[] for i in range(NUM_ITERS)]
        filter_imgs_loss = [[] for i in range(NUM_ITERS)]
        nb_filter = c.shape[-1]
        numIt = 0
        for numIt in range(NUM_ITERS):
        #for numIt in range(2):
#            imgList = []
#            imgLossList = []    
            print("------NumIt: " + str(numIt) + "---------" )
           
            for filter_idx in range(nb_filter):
                start_time = time.time()
                if numIt == 0:
                    #input_img_data = photo.reshape(1,48,48,1)
                    input_img_data = np.random.random((1, 48, 48, 1))
                else:
#                    input_img_data = filter_imgs[numIt-1][0][filter_idx]
                    input_img_data = filter_imgs[numIt-1][filter_idx].reshape(1,48,48,1)
                loss = K.mean(c[:,:,:,filter_idx])
                grads = normalize(K.gradients(loss,input_img)[0])
                iterate = K.function([input_img,K.learning_phase()],[loss,grads])
                tempImage, tempLoss = grad_ascent(NUM_STEPS, input_img_data, iterate)
                #print(filter_idx)
#                imgList.append(tempImage)
#                imgLossList.append(tempLoss)
                filter_imgs[numIt].append(deprocess_image(tempImage[0]))
#                filter_imgs[numIt].append(tempImage)
                filter_imgs_loss[numIt].append(tempLoss)
                end_time = time.time()
                print('NumIt %d -- Filter %d processed in %ds' % (numIt, filter_idx, end_time - start_time))
#            filter_imgs[numIt].append(imgList)
#            filter_imgs_loss[numIt].append(imgLossList)
            print(len(filter_imgs[numIt]))

        for it in range(NUM_ITERS):
        #for it in range(2):
            fig = plt.figure(figsize=(14,8))
            for i in range(nb_filter):
               # print(i)
                ax = fig.add_subplot(int(nb_filter)/16,16,i+1)
#                ax.imshow(filter_imgs[it][0][i].reshape(48,48),cmap='Blues')
                ax.imshow(filter_imgs[it][i].reshape(48,48),cmap='Blues')
                plt.xticks(np.array([]))
                plt.yticks(np.array([]))
#                plt.xlabel('{:.3f}'.format(filter_imgs_loss[it][0][i]))
                plt.xlabel('{:.3f}'.format(filter_imgs_loss[it][i]))
                plt.tight_layout()
            fig.suptitle('Filters of layer {} (# Ascent Epoch {} )'.format(name_ls[0],(it+1)))
            stringPath = "result/Visualizing_Filters_" + str(name_ls[0]) + "_" + str((it+1))
            fig.savefig(stringPath, dpi=100)

