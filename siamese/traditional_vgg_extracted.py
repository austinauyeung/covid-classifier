import glob
import numpy as np
import os
import matplotlib.pyplot as plt
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
cov = glob.glob(os.getcwd()+'/covid-chestxray-dataset-master/output/*')
pne = glob.glob(os.getcwd()+'/pneumonia2/*')

# parameters
cov_train_num = 100
pne_train_num = 100

# select random subset of images for training
cov_train = np.random.choice(cov,size=cov_train_num,replace=False)
pne_train = np.random.choice(pne,size=pne_train_num,replace=False)
cov_test = list(set(cov)-set(cov_train))
pne_test = list(set(pne)-set(pne_train))

cov_test_num = len(cov_test)
pne_test_num = len(pne_test)

print('COVID training set size: '+str(cov_train_num))
print('Pneumonia training set size: '+str(pne_train_num))
print('Total training set size: '+str(cov_train_num+pne_train_num))
print()
print('COVID testing set size: '+str(cov_test_num))
print('Pneumonia testing set size: '+str(pne_test_num))
print('Total testing set size: '+str(cov_test_num+pne_test_num))
print()

IMG_WIDTH = 224
IMG_HEIGHT = 224
IMG_DIM = (IMG_WIDTH,IMG_HEIGHT)

# load training images
cov_train_imgs = [img_to_array(load_img(img,target_size=IMG_DIM,color_mode="rgb")) for img in cov_train]
pne_train_imgs = [img_to_array(load_img(img,target_size=IMG_DIM,color_mode="rgb")) for img in pne_train]

# create corresponding labels
train_imgs = np.array(cov_train_imgs+pne_train_imgs)
train_imgs_scaled = train_imgs.astype('float32')/255
train_labels = np.array(cov_train_num*[1]+pne_train_num*[0])
train_labels_enc = to_categorical(train_labels)

# load test images and create corresponding labels
cov_test_imgs = [img_to_array(load_img(img,target_size=IMG_DIM,color_mode="rgb")) for img in cov_test]
pne_test_imgs = [img_to_array(load_img(img,target_size=IMG_DIM,color_mode="rgb")) for img in pne_test]
test_imgs = np.array(cov_test_imgs+pne_test_imgs)
test_imgs_scaled = test_imgs.astype('float32')/255
test_labels = np.array(cov_test_num*[1]+pne_test_num*[0])
test_labels_enc = to_categorical(test_labels)

# load weights from siamese network
adm = optimizers.Adam(lr=0.0001)
vgg_siamese = load_model('weights_vggtwin_tr19900_ep10.h5',compile=False)
vgg_siamese.compile(loss=hp.contrastive_loss, optimizer=adm, metrics=[hp.accuracy])

# create traditional classifier
input_shape = (IMG_HEIGHT,IMG_WIDTH,train_imgs.shape[3])
extracted_network = vgg_siamese.get_layer('model_1')
extracted_network.trainable = False
input = Input(shape=input_shape)
x = extracted_network(input)
# x = Dense(64, activation='relu')(x)
# x = Dropout(0.1)(x)
x = BatchNormalization()(x)
x = Dense(2, activation='softmax')(x)
vgg_traditional = Model(input,x)

# train model
epochs = 100;
adm = optimizers.Adam(lr=0.0001)
vgg_traditional.compile(loss='categorical_crossentropy', optimizer=adm, metrics=['accuracy'])
vgg_traditional.summary()
vgg_traditional.fit(train_imgs_scaled,train_labels_enc,
	batch_size=30,
	epochs=epochs,
	# validation_data=(test_imgs_scaled,test_labels_enc),
	shuffle=True)

train_labels_pred = vgg_traditional.predict(train_imgs_scaled)
train_acc = np.mean(train_labels==np.argmax(train_labels_pred,axis=1))
test_labels_pred = vgg_traditional.predict(test_imgs_scaled)
test_acc = np.mean(test_labels==np.argmax(test_labels_pred,axis=1))
print('Predicted labels:')
print(test_labels_pred)
print()
print('* Accuracy on training set: %0.2f%%' % (100 * train_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * test_acc))
print()

tp,tn,fp,fn,sensitivity,specificity = hp.generate_metrics(test_labels,np.argmax(test_labels_pred,axis=1))
print('True positives: '+str(tp))
print('True negatives: '+str(tn))
print('False positives: '+str(fp))
print('False negatives: '+str(fn))
print('Sensitivity: '+str(sensitivity))
print('Specificity: '+str(specificity))
