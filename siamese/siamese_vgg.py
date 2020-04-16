import glob
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import keras
import tensorflow as tf
import helpers as hp
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from sklearn.preprocessing import LabelEncoder
from keras.layers import Input, Lambda
from keras.models import Model, load_model
from keras import optimizers
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
cov_avg_num = 5

num_classes = 2

# select random subset of images for training
cov_train = np.random.choice(cov,size=cov_train_num,replace=False)
pne_train = np.random.choice(pne,size=pne_train_num,replace=False)
cov_test = list(set(cov)-set(cov_train))
pne_test = list(set(pne)-set(pne_train))
cov_avg = np.random.choice(cov_test,size=cov_avg_num,replace=False)
cov_test = list(set(cov_test)-set(cov_avg))

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
print('Number of COVID pictures to average for classifier: '+str(cov_avg_num))
print()
print('Maximum training pairs size:')
n = min([cov_train_num,pne_train_num])
n1 = cov_train_num
n2 = pne_train_num
print('     With class balance: '+str(2*n**2-n))
print('     Without class balance: '+str((n1*(n1-1))/2+(n2*(n2-1))/2+n1*n2))
print('Maximum testing pairs size:')
n = min([cov_test_num,pne_test_num])
n1 = cov_test_num
n2 = pne_test_num
print('     With class balance: '+str(2*n**2-n))
print('     Without class balance: '+str((n1*(n1-1))/2+(n2*(n2-1))/2+n1*n2))
print('Maximum classification pairs size:')
print('     With class balance: '+str(n*2))
print('     Without class balance: '+str(n1+n2))

# set following parameters to desired number of pairs according to output of above
train_balance = 1;
test_balance = 1;
cl_balance = 0;
num_tr_pairs = 19900;
num_te_pairs = 10878;

IMG_WIDTH = 224
IMG_HEIGHT = 224
IMG_DIM = (IMG_WIDTH,IMG_HEIGHT)

# load training images
cov_train_imgs = [img_to_array(load_img(img,target_size=IMG_DIM,color_mode="rgb")) for img in cov_train]
pne_train_imgs = [img_to_array(load_img(img,target_size=IMG_DIM,color_mode="rgb")) for img in pne_train]

# create corresponding labels
train_imgs = np.array(cov_train_imgs+pne_train_imgs)
train_imgs_scaled = train_imgs.astype('float32')/255
train_labels = cov_train_num*['c']+pne_train_num*['p']

# load test images and create corresponding labels
cov_test_imgs = [img_to_array(load_img(img,target_size=IMG_DIM,color_mode="rgb")) for img in cov_test]
pne_test_imgs = [img_to_array(load_img(img,target_size=IMG_DIM,color_mode="rgb")) for img in pne_test]
test_imgs = np.array(cov_test_imgs+pne_test_imgs)
test_imgs_scaled = test_imgs.astype('float32')/255
test_labels = cov_test_num*['c']+pne_test_num*['p']

# create average covid image
covavg_imgs = np.array([img_to_array(load_img(img,target_size=IMG_DIM,color_mode="rgb")) for img in cov_avg])
covavg_imgs_scaled = covavg_imgs.astype('float32')/255
if len(covavg_imgs_scaled.shape)>3:
    covavg_imgs_scaled = np.expand_dims(np.mean(covavg_imgs_scaled,axis=0),axis=0)
plt.title('Average COVID image')
plt.imshow(array_to_img(covavg_imgs_scaled[0]))
plt.show()

# encode class labels as 0/1
le = LabelEncoder()
le.fit(train_labels)
train_labels_enc = le.transform(train_labels)
test_labels_enc = le.transform(test_labels)

# create positive and negative pairs
idx = [np.where(train_labels_enc==i)[0] for i in range(num_classes)]
tr_pairs, tr_y = hp.create_all_pairs(num_classes,train_imgs_scaled,idx,train_balance)

idx = [np.where(test_labels_enc==i)[0] for i in range(num_classes)]
te_pairs, te_y = hp.create_all_pairs(num_classes,test_imgs_scaled,idx,test_balance)
cl_pairs, cl_y = hp.covavg_pairs(num_classes,covavg_imgs_scaled,test_imgs_scaled,idx,cl_balance)

# use random subset of all pairs (might need to adjust to ensure class balance)
idx = random.sample(range(tr_pairs.shape[0]),num_tr_pairs)
tr_pairs = np.array([tr_pairs[x] for x in idx])
tr_y = np.array([tr_y[x] for x in idx])

idx = random.sample(range(te_pairs.shape[0]),num_te_pairs)
te_pairs = np.array([te_pairs[x] for x in idx])
te_y = np.array([te_y[x] for x in idx])

print('Balancing training pairs: '+str(train_balance==1))
print('Balancing testing pairs: '+str(test_balance==1))
print('Balancing classification pairs: '+str(cl_balance==1))
print()
print('Training pairs size: '+str(tr_pairs.shape[0]))
print('Testing pairs size: '+str(te_pairs.shape[0]))
print('Classification pairs size: '+str(cl_pairs.shape[0]))

# create siamese network with euclidean distance as final layer
input_shape = (IMG_HEIGHT,IMG_WIDTH,train_imgs.shape[3])
base_network = hp.vggnet_base(input_shape)

input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(hp.euclidean_distance,output_shape=hp.eucl_dist_output_shape)([processed_a, processed_b])
model = Model([input_a, input_b], distance)
model.summary()

# train or load model
load = 1
adm = optimizers.Adam(lr=0.0001)
if load:
	model = load_model('weights_vggtwin_tr19900_ep10.h5',compile=False)
	model.compile(loss=hp.contrastive_loss, optimizer=adm, metrics=[hp.accuracy])
else:
	epochs = 10;
	model.compile(loss=hp.contrastive_loss, optimizer=adm, metrics=[hp.accuracy])
	model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
	          batch_size=128,
	          epochs=epochs,
	#           validation_data=([te_pairs[:10, 0], te_pairs[:10, 1]], te_y[:10]), # validate on small subset of testing dataset
	          shuffle=True)

	model.save('weights_vggtwin_tr'+str(tr_pairs.shape[0])+'_ep'+str(epochs)+'.h5')

tr_y_dist = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
tr_acc = hp.compute_accuracy(tr_y, tr_y_dist)
te_y_dist = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
te_acc = hp.compute_accuracy(te_y, te_y_dist)
cl_y_dist = model.predict([cl_pairs[:, 0], cl_pairs[:, 1]])
cl_acc = hp.compute_accuracy(cl_y, cl_y_dist)

print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
print('* Accuracy on classification set: %0.2f%%' % (100 * cl_acc))

threshold = 1
cl_y_pred = hp.generate_label(cl_y_dist,threshold) # using half of margin for threshold
tp,tn,fp,fn,sensitivity,specificity = hp.generate_metrics(cl_y,cl_y_pred)
print('True positives: '+str(tp))
print('True negatives: '+str(tn))
print('False positives: '+str(fp))
print('False negatives: '+str(fn))
print('Sensitivity: '+str(sensitivity))
print('Specificity: '+str(specificity))