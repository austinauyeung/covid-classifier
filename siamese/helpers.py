import numpy as np
import random
import os
import lime
import keras
import tensorflow as tf
from lime import lime_image
from itertools import combinations, product
from keras.applications import vgg16, resnet50, inception_v3
from keras.layers import Lambda, Dense, Dropout, Flatten, AveragePooling2D
from keras import backend as K
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from scipy.ndimage.interpolation import zoom
from skimage.segmentation import mark_boundaries
import cv2

# parameters
cov_train_num = 90
pne_train_num = 90
hea_train_num = 90
source_cov = os.getcwd()+'/sorted_files/covid/'
source_pne = os.getcwd()+'/sorted_files/pneumonia/'
source_hea = os.getcwd()+'/sorted_files/healthy/'
margin = 2
threshold = 1
batch_size = 128
alpha_heatmap = 0.005

# functions
def vggnet_base(input_shape):
    # exclude top 3 fully-connected layers and train last 5 of remaining layers
    vggnet = vgg16.VGG16(include_top=False,weights='imagenet',input_shape=input_shape,pooling='avg')
    for layer in vggnet.layers[-5:]:
        layer.trainable=True
    for layer in vggnet.layers[:-5]:
        layer.trainable=False
    x = vggnet.output

    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)

    return Model(vggnet.input,x)

def resnet_base(input_shape):
    resnet = resnet50.ResNet50(include_top=False,weights='imagenet',input_shape=input_shape,pooling='avg')
    for layer in resnet.layers[-15:]:
        layer.trainable=True
    for layer in resnet.layers[:-15]:
        layer.trainable=False
    x = resnet.output

    return Model(resnet.input,x)

def inception_base(input_shape):
    inception = inception_v3.InceptionV3(include_top=False,weights='imagenet',input_shape=input_shape,pooling='avg')
    for layer in inception.layers[-15:]:
        layer.trainable=True
    for layer in inception.layers[:-15]:
        layer.trainable=False
    x = inception.output

    return Model(inception.input,x)

datagen = ImageDataGenerator(
    fill_mode='constant',
    cval=0,
    rescale=1./1,
    featurewise_center=False,
    featurewise_std_normalization=False,
    samplewise_center=False,
    samplewise_std_normalization=False,
    rotation_range=15,
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=True,
    vertical_flip=False)

def trainGenerator(X,I,Y):
    while True:
        idx = np.random.permutation(X.shape[0])
        batches = datagen.flow(X[idx], Y[idx], batch_size=batch_size, shuffle=False)
        idx0 = 0
        for batch in batches:
            idx1 = idx0 + batch[0].shape[0]
            yield [batch[0], I[idx[idx0:idx1]]], batch[1]
            idx0 = idx1
            if idx1 >= X.shape[0]:
                break

def trainGenerator2(X1, X2, Y):
    genX1 = datagen.flow(X1, Y, seed=7, batch_size=batch_size)
    genX2 = datagen.flow(X2, seed=7, batch_size=batch_size)
    while True:
            X1i = genX1.next()
            X2i = genX2.next()
            yield [X1i[0],X2i], X1i[1]

def create_all_pairs(num_classes, x, class_indices, balance):
    '''Positive and negative pair creation.
    Use "balance" to toggle positive/negative pair balance.
    '''
    pairs = []
    labels = []

    # create all positive pairs for each class (two classes: (N choose 2)*2=N^2-N)
    for d in range(num_classes):
        if balance:
            n = min([len(class_indices[d]) for d in range(num_classes)])
        else:
            n = len(class_indices[d])
        comb = list(combinations(range(0,n),2)) # select indices in class
        for i in comb:
            z1, z2 = class_indices[d][i[0]], class_indices[d][i[1]]
            pairs += [[x[z1], x[z2]]]
            labels += [1]
    
    # create all possible negative pairs (two classes: N^2)
    comb = list(combinations(range(0,num_classes),2)) # select two different classes
    for d in comb:
        if balance:
            n0 = n
            n1 = n
        else:
            n0 = len(class_indices[d[0]])
            n1 = len(class_indices[d[1]])
        comb2 = list(product(range(n0),range(n1)))
        for e in comb2:
            z1, z2 = class_indices[d[0]][e[0]], class_indices[d[1]][e[1]]
            pairs += [[x[z1], x[z2]]]
            labels += [0]

    return np.array(pairs), np.array(labels)

def covavg_pairs(num_classes,x1,x2,class_indices,balance):
    '''Assumes two classes. Assumes class 0 of x2 matches x1.'''
    pairs = []
    labels = []
    n = min([len(class_indices[d]) for d in range(num_classes)])
    if balance:
        n0 = n
        n1 = n
    else:
        n0 = len(class_indices[0])
        n1 = len(class_indices[1])
    
    # create all positive pairs for each class
    for i in range(n1):
        z = class_indices[1][i]
        pairs += [[x1[0], x2[z]]]
        labels += [1]
    
    # create all possible negative pairs
    for i in range(n0):
        z = class_indices[0][i]
        pairs += [[x1[0], x2[z]]]
        labels += [0] 
                
    return np.array(pairs), np.array(labels)

def generate_label(y_pred, threshold):
    '''Compute classification labels with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < threshold
    return pred

def generate_metrics(y_true,y_pred):
    tp = sum(y_true+y_pred==2)
    tn = sum(y_true+y_pred==0)
    fp = sum(y_true-y_pred==-1)
    fn = sum(y_true-y_pred==1)
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)
    
    return np.array([tp,tn,fp,fn,sensitivity,specificity])

def extract_features(model,img,name):
    # LIME
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(img[0], model.predict, top_labels=5, hide_color=0, num_samples=1000)
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
    # plt.figure()
    # plt.imshow(mark_boundaries(temp, mask))
    marked = mark_boundaries(temp,mask)
    marked = np.uint8(255 * marked / np.max(marked))
    cv2.imwrite('img_'+name+'_lime.jpg',marked)

    # Grad-CAM
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)
    cam, heatmap = grad_cam(model, img, predicted_class, 'block5_conv3')
    cv2.imwrite('img_'+name+'_gradcam.jpg', cam)

    # Grad-CAM++
    gradcamplus=grad_cam_plus(model,img,layer_name='block5_conv3')
    cv2.imwrite('img_'+name+'_gradcamplus.jpg', gradcamplus)
    return marked, cv2.cvtColor(cam, cv2.COLOR_BGR2RGB), cv2.cvtColor(gradcamplus, cv2.COLOR_BGR2RGB)

# interpretable machine learning/feature extraction functions
# referenced from the following:
# LIME: https://github.com/marcotcr/lime
# Grad-CAM: https://github.com/jacobgil/keras-grad-cam
# Grad-CAM++: https://github.com/totti0223/gradcamplusplus

# Grad-CAM:
def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))

def target_category_loss_output_shape(input_shape):
    return input_shape

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def grad_cam(input_model, image, category_index, layer_name):
    nb_classes = 2
    target_layer = lambda x: target_category_loss(x, category_index, nb_classes)

    x = input_model.layers[-1].output
    x = Lambda(target_layer, output_shape=target_category_loss_output_shape)(x)
    model = keras.models.Model(input_model.layers[0].input, x)

    loss = K.sum(model.layers[-1].output)
    #conv_output = [l for l in model.layers[0].layers if l.name is layer_name][0].output
    conv_output = [l for l in model.layers if l.name == layer_name][0].output

    grads = normalize(K.gradients(loss, conv_output)[0])
    gradient_function = K.function([model.layers[0].input], [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis = (0, 1))
    cam = np.ones(output.shape[0 : 2], dtype = np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = cv2.resize(cam, (224, 224))
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)

    #Return to BGR [0..255] from the preprocessed image
    image = image[0, :]
    # plt.show()
    image -= np.min(image)
    image = np.minimum(image, 255)

    cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    cam = alpha_heatmap*np.float32(cam) + np.float32(image)
    cam = 255 * cam / np.max(cam)
    return np.uint8(cam), heatmap

# Grad-CAM++:
def grad_cam_plus(input_model, img, layer_name,H=224,W=224):
    cls = np.argmax(input_model.predict(img))
    y_c = input_model.output[0, cls]
    conv_output = input_model.get_layer(layer_name).output
    grads = K.gradients(y_c, conv_output)[0]

    first = K.exp(y_c)*grads
    second = K.exp(y_c)*grads*grads
    third = K.exp(y_c)*grads*grads*grads

    gradient_function = K.function([input_model.input], [y_c,first,second,third, conv_output, grads])
    y_c, conv_first_grad, conv_second_grad,conv_third_grad, conv_output, grads_val = gradient_function([img])
    global_sum = np.sum(conv_output[0].reshape((-1,conv_first_grad[0].shape[2])), axis=0)

    alpha_num = conv_second_grad[0]
    alpha_denom = conv_second_grad[0]*2.0 + conv_third_grad[0]*global_sum.reshape((1,1,conv_first_grad[0].shape[2]))
    alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, np.ones(alpha_denom.shape))
    alphas = alpha_num/alpha_denom

    weights = np.maximum(conv_first_grad[0], 0.0)

    alpha_normalization_constant = np.sum(np.sum(alphas, axis=0),axis=0)

    alphas /= alpha_normalization_constant.reshape((1,1,conv_first_grad[0].shape[2]))

    deep_linearization_weights = np.sum((weights*alphas).reshape((-1,conv_first_grad[0].shape[2])),axis=0)
    grad_CAM_map = np.sum(deep_linearization_weights*conv_output[0], axis=2)

    # Passing through ReLU
    cam = np.maximum(grad_CAM_map, 0)
    cam = zoom(cam,H/cam.shape[0])
    heatmap = cam / np.max(cam) # scale 0 to 1.0    

    #Return to BGR [0..255] from the preprocessed image
    img = img[0, :]
    img -= np.min(img)
    img = np.minimum(img, 255)

    cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    cam = alpha_heatmap*np.float32(cam) + np.float32(img)
    cam = 255 * cam / np.max(cam)
    return np.uint8(cam)

# following functions referenced from the following:
# https://github.com/keras-team/keras/blob/master/examples/mnist_siamese.py

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    # margin = 20
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < threshold
    return np.mean(pred == y_true)

def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < threshold, y_true.dtype)))

def create_pairs(x, class_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    n = min([len(class_indices[d]) for d in range(num_classes)]) - 1
    for d in range(num_classes):
        for i in range(n):
            z1, z2 = class_indices[d][i], class_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, num_classes)
            dn = (d + inc) % num_classes
            z1, z2 = class_indices[d][i], class_indices[dn][random.randrange(0,n+1)]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)