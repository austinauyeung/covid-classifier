import glob
import numpy as np
import os
import random
import keras
import tensorflow as tf
import helpers as hp
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from keras.layers import Input, Lambda, Dense, Dropout, BatchNormalization
from keras.models import Model, load_model
from keras import optimizers
from keras.utils import to_categorical
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
print("Number of GPUs available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print()

# set to image directory
cov = glob.glob(os.getcwd()+'/covid_processed/*')
pne = glob.glob(os.getcwd()+'/pne_processed/*')
hea = glob.glob(os.getcwd()+'/healthy_processed/*')

cov_uni = glob.glob(os.getcwd()+'/unique_processed/CO*')
pne_uni = glob.glob(os.getcwd()+'/unique_processed/Pn*')
hea_uni = glob.glob(os.getcwd()+'/unique_processed/No*')

# parameters
cov_train_num = 100
pne_train_num = 100
hea_train_num = 100

# select random subset of images for training
cov_train = np.random.choice(cov,size=cov_train_num,replace=False)
pne_train = np.random.choice(pne,size=pne_train_num,replace=False)
hea_train = np.random.choice(hea,size=hea_train_num,replace=False)
cov_test = list((set(cov)-set(cov_train))|set(cov_uni))
pne_test = list((set(pne)-set(pne_train))|set(pne_uni))
hea_test = list((set(hea)-set(hea_train))|set(hea_uni))

cov_test_num = len(cov_test)
pne_test_num = len(pne_test)
hea_test_num = len(hea_test)

print('Model 1: COVID vs. pneumonia:')
print('     COVID training set size: '+str(cov_train_num))
print('     Pneumonia training set size: '+str(pne_train_num))
print('     Total training set size: '+str(cov_train_num+pne_train_num))
print()
print('     COVID testing set size: '+str(cov_test_num))
print('     Pneumonia testing set size: '+str(pne_test_num))
print('     Total testing set size: '+str(cov_test_num+pne_test_num))
print('Model 2: COVID vs. healthy:')
print('     COVID training set size: '+str(cov_train_num))
print('     Healthy training set size: '+str(hea_train_num))
print('     Total training set size: '+str(cov_train_num+hea_train_num))
print()
print('     COVID testing set size: '+str(cov_test_num))
print('     Healthy testing set size: '+str(hea_test_num))
print('     Total testing set size: '+str(cov_test_num+hea_test_num))
print()

IMG_WIDTH = 224
IMG_HEIGHT = 224
IMG_DIM = (IMG_WIDTH,IMG_HEIGHT)

# load training images and create corresponding labels
cov_train_imgs = [img_to_array(load_img(img,target_size=IMG_DIM,color_mode="rgb")) for img in cov_train]
pne_train_imgs = [img_to_array(load_img(img,target_size=IMG_DIM,color_mode="rgb")) for img in pne_train]
hea_train_imgs = [img_to_array(load_img(img,target_size=IMG_DIM,color_mode="rgb")) for img in hea_train]

mod1_train_imgs = np.array(cov_train_imgs+pne_train_imgs)
mod1_train_imgs = mod1_train_imgs.astype('float32')/255
mod1_train_labels = np.array(cov_train_num*[1]+pne_train_num*[0])
mod1_train_labels = to_categorical(mod1_train_labels)
mod2_train_imgs = np.array(cov_train_imgs+hea_train_imgs)
mod2_train_imgs = mod2_train_imgs.astype('float32')/255
mod2_train_labels = np.array(cov_train_num*[1]+hea_train_num*[0])
mod2_train_labels = to_categorical(mod2_train_labels)

# load test images and create corresponding labels
cov_test_imgs = [img_to_array(load_img(img,target_size=IMG_DIM,color_mode="rgb")) for img in cov_test]
pne_test_imgs = [img_to_array(load_img(img,target_size=IMG_DIM,color_mode="rgb")) for img in pne_test]
hea_test_imgs = [img_to_array(load_img(img,target_size=IMG_DIM,color_mode="rgb")) for img in hea_test]

mod1_test_imgs = np.array(cov_test_imgs+pne_test_imgs)
mod1_test_imgs = mod1_test_imgs.astype('float32')/255
mod1_test_labels = np.array(cov_test_num*[1]+pne_test_num*[0])
mod1_test_labels = to_categorical(mod1_test_labels)
mod2_test_imgs = np.array(cov_test_imgs+hea_test_imgs)
mod2_test_imgs = mod2_test_imgs.astype('float32')/255
mod2_test_labels = np.array(cov_test_num*[1]+hea_test_num*[0])
mod2_test_labels = to_categorical(mod2_test_labels)

# temporary: load unique images
cov_uni_imgs = np.array([img_to_array(load_img(img,target_size=IMG_DIM,color_mode="rgb")) for img in cov_uni])
cov_uni_imgs = cov_uni_imgs.astype('float32')/255
pne_uni_imgs = np.array([img_to_array(load_img(img,target_size=IMG_DIM,color_mode="rgb")) for img in pne_uni])
pne_uni_imgs = pne_uni_imgs.astype('float32')/255
hea_uni_imgs = np.array([img_to_array(load_img(img,target_size=IMG_DIM,color_mode="rgb")) for img in hea_uni])
hea_uni_imgs = hea_uni_imgs.astype('float32')/255

# create traditional classifier
input_shape = (IMG_HEIGHT,IMG_WIDTH,mod1_train_imgs.shape[3])
base_network = hp.vggnet_base(input_shape)
x = base_network.output
# x = Dense(64, activation='relu')(x)
# x = Dropout(0.1)(x)
x = BatchNormalization()(x)
x = Dense(2, activation='softmax')(x)
vgg_mod1 = Model(base_network.input,x)
vgg_mod2 = Model(base_network.input,x)

# train or load model
load = 0
adm = optimizers.Adam(lr=0.0001)
if load:
    vgg_mod1 = load_model('mod1_tr200_ep50.h5',compile=False)
    vgg_mod1.compile(loss='categorical_crossentropy', optimizer=adm, metrics=['accuracy'])
    vgg_mod2 = load_model('mod2_tr200_ep50.h5',compile=False)
    vgg_mod2.compile(loss='categorical_crossentropy', optimizer=adm, metrics=['accuracy'])
else:
    epochs = 50;
    vgg_mod1.compile(loss='categorical_crossentropy', optimizer=adm, metrics=['accuracy'])
    vgg_mod1.summary()
    vgg_mod1.fit(mod1_train_imgs,mod1_train_labels,
        batch_size=128,
        epochs=epochs,
        # validation_data=(test_imgs_scaled,test_labels_enc),
        shuffle=True)
    vgg_mod1.save('mod1_tr'+str(mod1_train_imgs.shape[0])+'_ep'+str(epochs)+'.h5')

    vgg_mod2.compile(loss='categorical_crossentropy', optimizer=adm, metrics=['accuracy'])
    vgg_mod2.summary()
    vgg_mod2.fit(mod2_train_imgs,mod2_train_labels,
        batch_size=128,
        epochs=epochs,
        # validation_data=(test_imgs_scaled,test_labels_enc),
        shuffle=True)
    vgg_mod2.save('mod2_tr'+str(mod2_train_imgs.shape[0])+'_ep'+str(epochs)+'.h5')

    mod1_train_labels_pred = vgg_mod1.predict(mod1_train_imgs)
    mod1_train_acc = np.mean(np.argmax(mod1_train_labels,axis=1)==np.argmax(mod1_train_labels_pred,axis=1))
    mod1_test_labels_pred = vgg_mod1.predict(mod1_test_imgs)
    mod1_test_acc = np.mean(np.argmax(mod1_test_labels,axis=1)==np.argmax(mod1_test_labels_pred,axis=1))
    tp,tn,fp,fn,sensitivity,specificity = hp.generate_metrics(np.argmax(mod1_test_labels,axis=1),np.argmax(mod1_test_labels_pred,axis=1))
    print('Predicted labels:')
    print(mod1_test_labels_pred)
    print('Model 1, COVID vs. pneumonia:')
    print('     Accuracy on training set: %0.2f%%' % (100 * mod1_train_acc))
    print('     Accuracy on test set: %0.2f%%' % (100 * mod1_test_acc))
    print()
    print('     True positives: '+str(tp))
    print('     True negatives: '+str(tn))
    print('     False positives: '+str(fp))
    print('     False negatives: '+str(fn))
    print('     Sensitivity: '+str(sensitivity))
    print('     Specificity: '+str(specificity))
    
    mod2_train_labels_pred = vgg_mod2.predict(mod2_train_imgs)
    mod2_train_acc = np.mean(np.argmax(mod2_train_labels,axis=1)==np.argmax(mod2_train_labels_pred,axis=1))
    mod2_test_labels_pred = vgg_mod2.predict(mod2_test_imgs)
    mod2_test_acc = np.mean(np.argmax(mod2_test_labels,axis=1)==np.argmax(mod2_test_labels_pred,axis=1))
    tp,tn,fp,fn,sensitivity,specificity = hp.generate_metrics(np.argmax(mod2_test_labels,axis=1),np.argmax(mod2_test_labels_pred,axis=1))
    print('Predicted labels:')
    print(mod2_test_labels_pred)
    print('Model 2, COVID vs. healthy:')
    print('     Accuracy on training set: %0.2f%%' % (100 * mod2_train_acc))
    print('     Accuracy on test set: %0.2f%%' % (100 * mod2_test_acc))
    print()
    print('     True positives: '+str(tp))
    print('     True negatives: '+str(tn))
    print('     False positives: '+str(fp))
    print('     False negatives: '+str(fn))
    print('     Sensitivity: '+str(sensitivity))
    print('     Specificity: '+str(specificity))