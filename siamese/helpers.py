import numpy as np
import random
from itertools import combinations, product
from keras.applications import vgg16
from keras.layers import Dense, Dropout
from keras import backend as K
from keras.models import Model

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

def create_all_pairs(num_classes, x, class_indices, balance):
    '''Positive and negative pair creation.
    Use "balance" to toggle positive/negative pair balance.
    Use "limit" to limit the number of pairs and conserve memory.
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
    for i in range(n0):
        z = class_indices[0][i]
        pairs += [[x1[0], x2[z]]]
        labels += [1]
    
    # create all possible negative pairs
    for i in range(n1):
        z = class_indices[1][i]
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
    margin = 2
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 1
    return np.mean(pred == y_true)

def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 1, y_true.dtype)))

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