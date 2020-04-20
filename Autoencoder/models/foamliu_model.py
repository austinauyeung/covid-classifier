import keras.backend as K
from keras.layers import Input, Conv2D, UpSampling2D, BatchNormalization, ZeroPadding2D, MaxPooling2D
from keras.models import Model
from keras.utils import plot_model

# Credits: https://github.com/foamliu/Conv-Autoencoder/blob/master/model.py
def create_model(input_shape):
    assert input_shape[0] % 32 == 0, "Input dimensions must be a multiple of 32"
    assert input_shape[1] % 32 == 0, "Input dimensions must be a multiple of 32"
    # Encoder 
    input_tensor = Input(input_shape)
    x = Conv2D(64, (3, 3), activation='relu', name='conv1_1', padding='same')(input_tensor)
    x = Conv2D(64, (3, 3), activation='relu', name='conv1_2', padding='same')(x)
    #print()
    #print("Shape before downsample 1: \t" + str(x.shape))
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', name='conv2_1', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', name='conv2_2', padding='same')(x)  
    #print("Shape before downsample 2: \t" + str(x.shape))
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Conv2D(256, (3, 3), activation='relu', name='conv3_1', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', name='conv3_2', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', name='conv3_3', padding='same')(x)
    #print("Shape before downsample 3: \t" + str(x.shape))
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Conv2D(512, (3, 3), activation='relu', name='conv4_1', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', name='conv4_2', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', name='conv4_3', padding='same')(x)
    #print("Shape before downsample 4: \t" + str(x.shape))
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Conv2D(512, (3, 3), activation='relu', name='conv5_1', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', name='conv5_2', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', name='conv5_3', padding='same')(x)  
    #print("Shape before downsample 5: \t" + str(x.shape))
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    #print("Shape of bottleneck vector: \t" + str(x.shape))
    # Decoder
    x = Conv2D(512, (1, 1), activation='relu', padding='same', name='deconv6', kernel_initializer='he_normal',
               bias_initializer='zeros')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D(size=(2, 2), interpolation='nearest')(x)
    #print("Shape after upsample 1: \t" + str(x.shape))
    x = Conv2D(512, (5, 5), activation='relu', padding='same', name='deconv5', kernel_initializer='he_normal',
               bias_initializer='zeros')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D(size=(2, 2), interpolation='nearest')(x)
    #print("Shape after upsample 2: \t" + str(x.shape))
    x = Conv2D(256, (5, 5), activation='relu', padding='same', name='deconv4', kernel_initializer='he_normal',
               bias_initializer='zeros')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D(size=(2, 2), interpolation='nearest')(x)
    #print("Shape after upsample 3: \t" + str(x.shape))
    x = Conv2D(128, (5, 5), activation='relu', padding='same', name='deconv3', kernel_initializer='he_normal',
               bias_initializer='zeros')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D(size=(2, 2), interpolation='nearest')(x)
    #print("Shape after upsample 4: \t" + str(x.shape))
    x = Conv2D(64, (5, 5), activation='relu', padding='same', name='deconv2', kernel_initializer='he_normal',
               bias_initializer='zeros')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D(size=(2, 2), interpolation='nearest')(x)
    #print("Shape after upsample 5: \t" + str(x.shape))
    #print()
    x = Conv2D(64, (5, 5), activation='relu', padding='same', name='deconv1', kernel_initializer='he_normal',
               bias_initializer='zeros')(x)
    x = BatchNormalization()(x)
    x = Conv2D(1, (5, 5), activation='sigmoid', padding='same', name='pred', kernel_initializer='he_normal',
               bias_initializer='zeros')(x)
    model = Model(inputs=input_tensor, outputs=x)
    return model

if __name__ == "__main__":
    print(create_model((320, 384, 1)).summary())