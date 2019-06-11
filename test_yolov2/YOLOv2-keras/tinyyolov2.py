"""Model file for TinyYolo v2."""

from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization, Reshape, Input, Lambda, Activation
from keras.layers import LeakyReLU
from keras.layers.merge import concatenate

import tensorflow as tf

class TinyYolov2():
    def __init__(self, input_shape):
        input = Input(input_shape)
        #model = Sequential()
        
        # Layer 1
        x = Conv2D(16, (3,3), strides=(1,1), padding='same', use_bias=False)(input)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
        # Layer 2 - 5
        for i in range(0,4):
            x = Conv2D(32*(2**i), (3,3), strides=(1,1), padding='same', use_bias=False)(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(alpha=0.1)(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
        
        # Layer 6
        x = Conv2D(512, (3,3), strides=(1,1), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(x)
        
        # Layer 7
        x = Conv2D(1024, (3,3), strides=(1,1), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        
        # Layer 8
        x = Conv2D(512, (3,3), strides=(1,1), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        
        # Layer 9
        out = Conv2D(5 * (4 + 1 + 80), (1, 1), strides=(1, 1), padding='same', name='conv9')(x)

        self.model = Model(inputs = input, outputs = out)
        self.model.summary()

    def get_model(self):
        return self.model
