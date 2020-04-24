import numpy as np
import pandas as pd
import cv2
import os

from glob import glob
from collections import defaultdict
from random import choice

from keras.preprocessing import image
from keras_vggface.utils import preprocess_input

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Input, Dense, GlobalMaxPool2D, GlobalAvgPool2D, Concatenate, Multiply, Dropout, Subtract
from keras.models import Model
from keras.optimizers import Adam

from keras_vggface.vggface import VGGFace

def get_dataset(filename):
    dataset = defaultdict(list)
    class_name = None
    with open(filename, 'r') as datafile:
        lines = datafile.readlines()
        for line in lines:
            if '#' in line:
                class_name = line[1:-1]
            else:
                dataset[class_name].append(line[:-1])
    return dataset

train_class_to_img_map = get_dataset("train.txt")
test_class_to_img_map = get_dataset("test.txt")
classes = list(train_class_to_img_map.keys())

print("\n\n======================================== DATA SET INFO. =========================================\n")
print("Classes Found: " + str(classes))
print("Number of Train Images Found: " + str(sum([len(x) for x in train_class_to_img_map.values()])))
print("Number of Test Images Found:  " + str(sum([len(x) for x in test_class_to_img_map.values()])))
print("\n==================================================================================================\n")


def read_img(path):
    img = image.load_img(path, target_size=(197, 197))
    img = np.array(img).astype(np.float)
    return preprocess_input(img, version=2)

def gen(class_to_img_map, batch_size=16):
    while True:
        batch_tuples = []
        labels = []
        while len(batch_tuples) < batch_size:
            c1 = choice(classes)
            c2 = choice(classes)
            batch_tuples.append((c1, c2))
            labels.append(int(c1 == c2))
        
        X1 = [choice(class_to_img_map[x[0]]) for x in batch_tuples]
        X1 = np.array([read_img(x) for x in X1])

        X2 = [choice(class_to_img_map[x[1]]) for x in batch_tuples]
        X2 = np.array([read_img(x) for x in X2])

        yield [X1, X2], labels
            
def baseline_model():
    input_1 = Input(shape=(197, 197, 3))
    input_2 = Input(shape=(197, 197, 3))

    base_model = VGGFace(model='resnet50', include_top=False)

    # Make trainable only the last 3 layers
    for x in base_model.layers[-3:]:
        x.trainable = True

    for x in base_model.layers[:-3]:
        x.trainable = False

    x1 = base_model(input_1)
    x2 = base_model(input_2)

    x1 = Concatenate(axis=-1)([GlobalMaxPool2D()(x1), GlobalAvgPool2D()(x1)])
    x2 = Concatenate(axis=-1)([GlobalMaxPool2D()(x2), GlobalAvgPool2D()(x2)])

    x3 = Subtract()([x1, x2])
    x3 = Multiply()([x3, x3])

    x1_ = Multiply()([x1, x1])
    x2_ = Multiply()([x2, x2])
    x4 = Subtract()([x1_, x2_])
    x = Concatenate(axis=-1)([x4, x3])

    x = Dense(100, activation="relu")(x)
    x = Dropout(0.01)(x)
    out = Dense(1, activation="sigmoid")(x)

    model = Model([input_1, input_2], out)

    model.compile(loss="binary_crossentropy", metrics=['acc'], optimizer=Adam(0.00001))

    model.summary()

    return model

def main():
    file_path = "weights.h5"

    checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    reduce_on_plateau = ReduceLROnPlateau(monitor="val_acc", mode="max", factor=0.1, patience=20, verbose=1)

    callbacks_list = [checkpoint, reduce_on_plateau]

    model = baseline_model()

    if os.path.exists(file_path):
        model.load_weights(file_path)

    model.fit_generator(gen(train_class_to_img_map, batch_size=16), validation_data=gen(test_class_to_img_map,\
                        batch_size=16), epochs=100, verbose=2, callbacks=callbacks_list, steps_per_epoch=100, \
                        validation_steps=100)

if __name__ == '__main__':
    main()