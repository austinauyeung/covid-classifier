import numpy as np

from os import listdir
from os.path import isfile, join, exists

from random import sample

from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.optimizers import Adam

from models.foamliu_model import create_model

te_co_pids = [str(i) for i in range(150, 200)]
te_no_pids = [str(i) for i in range(1200, 1336)]
te_pn_pids = [str(i) for i in range(3500, 3917)]
te_pids = te_co_pids + te_no_pids + te_pn_pids

dirname = "images"
dataset = [f for f in listdir(dirname) if isfile(join(dirname, f)) and not f.split('_')[1] in te_pids]
print("Number of Train Images Found: " + str(len(dataset)))

def read_img(file):
    img = image.load_img(join(dirname, file), target_size=(320, 384), color_mode="grayscale")
    img = np.array(img).astype(np.float)
    img = img.reshape(list(img.shape) + [1])
    return img

def gen(dataset, batch_size=16):
    while True:
        batch = sample(dataset, batch_size)
        X = np.array([read_img(x) for x in batch])
        yield X, X

def main():
    input_shape = (320, 384, 1)
    weights_file = "weights.h5"

    model = create_model(input_shape)
    model.compile(loss="mean_squared_error", optimizer="sgd")

    if exists(weights_file):
        print("The file " + str(weights_file) + " will be overwritten. Do you want to continue? (y/n): ")
        usr_in = input()
        if usr_in.lower() != "y":
            print("Aborting")
            return
        model.load_weights(weights_file)
    
    callbacks_list = [ModelCheckpoint(weights_file, save_weights_only=False)]    
    model.fit_generator(gen(dataset, batch_size=16), epochs=100, verbose=2, callbacks=callbacks_list, steps_per_epoch=100)

if __name__ == "__main__":
    main()