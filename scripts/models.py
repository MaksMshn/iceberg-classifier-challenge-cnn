#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is describing a ConvNet model, with its parameters could be set 
at 'param.py'. It takes multi-inputs which are TWO-channels and meta information 
such as 'inc_angle'.

@author: cttsai (Chia-Ta Tsai), @Oct 2017
"""
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten
from keras.layers import GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.optimizers import Adam


def conv_block(x, nf=8, k=3, s=1, nb=2, p_act='elu'):
    
    for i in range(nb):
        x = Conv2D(filters=nf, kernel_size=(k, k), strides=(s, s),  
                   activation=p_act,
                   padding='same', kernel_initializer='he_uniform')(x)
        
    return x

def dense_block(x, h=32, d=0.5, m=0., p_act='elu'):
    return Dropout(d) (BatchNormalization(momentum=m) (Dense(h, activation=p_act)(x)))


def bn_pooling(x, k=2, s=2, m=0): 
    return MaxPooling2D((k, k), strides=(s, s))(BatchNormalization(momentum=m)(x))
    

############################################################################
# genome script    
############################################################################


import numpy as np
import random as rand
import math
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization


class GenomeHandler:
    def __init__(self, max_conv_layers, max_dense_layers, max_filters, max_dense_nodes,
                input_shape, n_classes, batch_normalization=True, dropout=True, max_pooling=True,
                optimizers=None, activations=None):
        if max_dense_layers < 1:
            raise ValueError("At least one dense layer is required for softmax layer") 
        filter_range_max = int(math.log(max_filters, 2)) + 1 if max_filters > 0 else 0
        self.optimizer = optimizers or [
            'adam',
            'rmsprop',
            'adagrad',
            'adadelta'
        ]
        self.activation = activations or [
            'relu',
            'sigmoid',
        ]
        self.convolutional_layer_shape = [
            "active",
            "num filters",
            "batch normalization",
            "activation",
            "dropout",
            "max pooling",
        ]
        self.dense_layer_shape = [
            "active",
            "num nodes",
            "batch normalization",
            "activation",
            "dropout",
        ]
        self.layer_params = {
            "active": [0, 1],
            "num filters": [2**i for i in range(3, filter_range_max)],
            "num nodes": [2**i for i in range(4, int(math.log(max_dense_nodes, 2)) + 1)],
            "batch normalization": [0, (1 if batch_normalization else 0)],
            "activation": list(range(len(self.activation))),
            "dropout": [(i if dropout else 0) for i in range(11)],
            "max pooling": list(range(3)) if max_pooling else 0,
        }

        self.convolution_layers = max_conv_layers
        self.convolution_layer_size = len(self.convolutional_layer_shape)
        self.dense_layers = max_dense_layers - 1 # this doesn't include the softmax layer, so -1
        self.dense_layer_size = len(self.dense_layer_shape)
        self.input_shape = input_shape
        self.n_classes = n_classes

    def convParam(self, i):
        key = self.convolutional_layer_shape[i]
        return self.layer_params[key]

    def denseParam(self, i):
        key = self.dense_layer_shape[i]
        return self.layer_params[key]

    def decode(self, genome):
        if not self.is_compatible_genome(genome):
            raise ValueError("Invalid genome for specified configs")
        model = Sequential()
        offset = 0
        dim = min(self.input_shape[:-1]) # keep track of smallest dimension
        input_layer = True
        for i in range(self.convolution_layers):
            if genome[offset]:
                convolution = None
                if input_layer:
                    convolution = Convolution2D(
                                        genome[offset + 1], (3, 3),
                                        padding='same',
                                        input_shape=self.input_shape)
                    input_layer = False
                else:
                    convolution = Convolution2D(
                                        genome[offset + 1], (3, 3),
                                        padding='same')
                model.add(convolution)
                if genome[offset + 2]:
                    model.add(BatchNormalization())
                model.add(Activation(self.activation[genome[offset + 3]]))
                model.add(Dropout(float(genome[offset + 4] / 20.0)))
                max_pooling_type = genome[offset + 5]
                # must be large enough for a convolution
                if max_pooling_type == 1 and dim >= 5:
                    model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
                    dim = int(math.ceil(dim / 2))
            offset += self.convolution_layer_size

        if not input_layer:
            model.add(Flatten())

        for i in range(self.dense_layers):
            if genome[offset]:
                dense = None
                if input_layer:
                    dense = Dense(genome[offset + 1], input_shape=self.input_shape)
                    input_layer = False
                else:
                    dense = Dense(genome[offset + 1])
                model.add(dense)
                if genome[offset + 2]:
                    model.add(BatchNormalization())
                model.add(Activation(self.activation[genome[offset + 3]]))
                model.add(Dropout(float(genome[offset + 4] / 20.0)))
            offset += self.dense_layer_size
        
        model.add(Dense(self.n_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
            optimizer=self.optimizer[genome[offset]],
            metrics=["accuracy"])
        return model
    
    
    def is_compatible_genome(self, genome):
        expected_len = self.convolution_layers * self.convolution_layer_size \
                        + self.dense_layers * self.dense_layer_size + 1
        if len(genome) != expected_len:
            return False
        ind = 0
        for i in range(self.convolution_layers):
            for j in range(self.convolution_layer_size):
                if genome[ind + j] not in self.convParam(j):
                    return False
            ind += self.convolution_layer_size
        for i in range(self.dense_layers):
            for j in range(self.dense_layer_size):
                if genome[ind + j] not in self.denseParam(j):
                    return False
            ind += self.dense_layer_size
        if genome[ind] not in range(len(self.optimizer)):
            return False
        return True
        
max_conv_layers = 6
max_dense_layers = 2
max_conv_kernels=256
max_dense_nodes = 1024

gh = GenomeHandler(max_conv_layers, max_dense_layers, max_conv_kernels, max_dense_nodes, 
                   input_shape=(75, 75, 2), n_classes=2)

max_conv_layers = 8
max_dense_layers = 3
max_conv_kernels=256
max_dense_nodes = 1024

gh2 = GenomeHandler(max_conv_layers, max_dense_layers, max_conv_kernels, max_dense_nodes, 
                   input_shape=(75, 75, 2), n_classes=2)

txt = """Fri Dec 29 14:11:35 2017.csv:1,256,1,0,0,1,1,16,1,1,5,1,0,8,1,1,0,1,0,64,1,0,0,2,1,256,1,0,5,2,1,64,0,1,7,0,1,128,1,1,3,2,0.25565173762,0.909395974754
Fri Dec 29 14:11:35 2017.csv:1,256,1,0,0,1,1,16,1,1,5,1,0,16,1,1,7,1,1,64,1,0,0,2,1,256,1,0,5,2,1,16,0,1,7,0,1,128,1,1,2,3,0.280610015728,0.906040269256
Fri Dec 29 14:11:35 2017.csv:1,256,1,0,0,1,1,16,1,1,2,1,0,8,1,1,0,1,0,64,1,0,0,2,1,256,1,0,5,2,1,64,0,1,7,0,1,128,1,1,2,3,0.284314403798,0.906040268856
Fri Dec 29 14:11:35 2017.csv:1,256,1,0,0,1,1,16,1,1,5,1,1,16,0,0,2,2,1,32,1,0,7,1,0,16,0,0,5,2,1,64,0,1,10,0,1,128,1,1,9,3,0.288482445198,0.872483221877
Fri Dec 29 14:11:35 2017.csv:1,256,1,0,0,1,1,16,1,1,2,1,0,8,1,1,0,1,0,64,1,0,0,2,1,256,1,0,5,2,1,64,0,0,7,0,1,512,1,1,2,3,0.271272370879,0.906040268856
Fri Dec 29 14:11:35 2017.csv:1,256,1,0,0,1,1,256,1,0,9,1,1,128,1,0,2,0,1,64,0,0,2,0,1,64,0,1,9,0,0,256,0,0,7,0,1,128,1,1,4,2,0.279337601374,0.912751677052
Fri Dec 29 14:11:35 2017.csv:1,256,1,0,0,1,1,16,1,0,2,1,0,8,1,1,0,1,0,64,1,0,0,2,1,256,1,0,5,2,1,64,0,1,7,0,1,128,1,0,2,3,0.255532039092,0.906040270057
Fri Dec 29 14:19:50 2017.csv:1,256,0,0,4,0,0,16,0,0,7,0,0,8,1,1,3,0,0,128,0,1,10,2,0,8,1,0,10,0,1,16,1,0,2,0,1,32,0,1,6,3,0.29026364119,0.902597401049"""


txt2 = """Sat Dec 30 01:35:52 2017.csv:1,64,1,0,3,2,1,32,1,0,4,0,1,64,0,1,4,1,0,256,0,1,6,2,1,256,0,1,2,1,1,8,1,0,8,1,0,16,0,0,8,1,0,32,0,0,1,0,1,32,0,0,5,1,128,1,1,4,2,0.287037402769,0.924050632911
Sat Dec 30 01:35:52 2017.csv:1,64,1,0,3,2,1,32,1,0,4,0,1,16,0,1,1,0,0,32,1,0,5,1,0,16,0,1,9,2,1,64,1,1,0,1,0,16,1,0,3,2,1,16,0,1,2,0,0,16,0,0,0,0,512,1,1,7,3,0.274483182008,0.917721523514
Sat Dec 30 01:35:52 2017.csv:1,256,1,0,3,2,1,32,1,0,4,0,0,16,1,0,1,0,0,32,1,0,5,1,0,16,0,1,9,2,1,64,0,1,0,1,0,16,1,0,3,2,1,16,0,1,2,0,0,16,0,0,0,0,512,1,1,7,3,0.277400854268,0.908227854137
Sat Dec 30 01:35:52 2017.csv:1,16,1,0,2,0,1,8,0,1,2,1,1,32,0,0,8,0,0,128,0,1,6,1,1,128,0,0,0,0,1,16,1,0,3,1,1,16,1,0,3,2,1,16,0,1,2,0,0,16,0,0,0,0,512,1,1,7,3,0.293233096222,0.901898738704
Sat Dec 30 01:35:52 2017.csv:1,16,1,0,2,0,1,8,0,1,2,1,1,32,0,0,8,0,0,128,0,1,6,1,1,128,0,0,0,0,1,16,1,0,3,1,1,16,1,0,3,2,1,16,0,1,2,0,0,16,0,0,0,0,512,1,1,7,3,0.293631664183,0.889240510856
Sat Dec 30 01:35:52 2017.csv:1,64,1,0,2,0,1,8,0,1,0,0,1,32,0,0,8,0,0,128,0,1,6,1,1,128,0,0,0,0,1,16,1,0,3,1,1,16,1,0,3,2,1,16,0,1,2,0,0,16,0,0,0,0,512,1,1,6,3,0.288561387153,0.889240509347
Sat Dec 30 01:35:52 2017.csv:1,16,1,0,2,0,1,8,0,1,0,1,1,32,0,0,8,0,0,128,0,1,6,1,1,128,0,0,8,0,1,16,1,0,3,1,0,16,1,0,3,2,1,64,0,1,2,0,0,16,0,0,0,0,512,1,1,7,0,0.286367276801,0.901898738704
Sat Dec 30 01:35:52 2017.csv:1,64,1,1,7,0,1,128,0,1,2,1,1,32,1,0,3,1,0,16,0,1,6,1,1,128,0,0,0,0,1,16,1,0,3,1,0,16,1,0,3,2,1,8,0,1,2,0,0,16,0,0,0,0,512,1,1,7,0,0.294597781912,0.898734181742
Sat Dec 30 01:35:52 2017.csv:1,64,1,0,0,0,1,8,0,1,2,2,1,32,1,0,0,1,0,16,0,1,6,1,1,8,0,0,0,2,1,16,1,0,3,1,1,128,1,1,10,2,1,16,1,1,2,0,0,16,0,0,0,0,512,1,1,7,0,0.294207079501,0.889240510856
Sat Dec 30 01:35:52 2017.csv:1,16,0,0,2,1,0,8,0,1,4,0,1,32,0,1,8,0,0,128,0,1,6,1,1,128,0,0,0,0,1,16,1,0,3,1,0,16,1,0,3,2,1,16,1,1,2,2,0,16,0,0,0,0,512,1,1,7,3,0.294493760866,0.908227851119
Sat Dec 30 01:35:52 2017.csv:1,16,1,0,1,2,1,8,0,1,2,0,1,32,1,0,3,1,0,16,0,1,6,1,0,128,0,0,0,0,1,32,0,0,3,1,1,128,0,0,3,2,1,16,0,1,6,2,0,16,0,0,0,1,16,1,1,7,3,0.285211491057,0.901898740213
Sat Dec 30 01:35:52 2017.csv:1,64,0,0,2,1,1,8,0,1,2,0,0,16,1,1,4,0,0,16,0,1,6,1,1,128,0,0,10,0,1,16,1,0,3,1,1,16,1,1,3,0,0,256,0,1,2,1,0,16,0,0,0,1,512,1,1,7,3,0.280875152614,0.901898735686
Sat Dec 30 01:35:52 2017.csv:1,16,0,0,2,1,0,8,0,1,4,0,1,32,0,1,8,2,0,128,0,1,6,1,1,128,1,0,0,0,1,16,1,0,3,1,1,64,1,0,10,2,1,16,0,1,2,1,0,16,0,0,0,1,16,1,1,7,0,0.289643395551,0.898734183251
Sat Dec 30 01:35:52 2017.csv:1,16,0,0,2,1,0,8,0,1,4,0,1,32,0,1,8,2,0,128,0,1,6,1,1,128,1,0,0,0,1,16,1,0,3,1,1,128,0,0,3,2,0,16,0,1,6,2,0,16,0,0,0,0,512,1,1,7,3,0.29119616596,0.901898735686
Sat Dec 30 01:35:52 2017.csv:1,256,0,0,2,1,1,8,0,0,2,0,1,32,0,0,2,0,0,16,0,1,6,1,0,128,0,1,4,0,1,16,1,0,1,1,0,128,0,0,10,2,1,128,0,0,2,1,0,16,0,0,0,0,512,1,1,7,3,0.299459516059,0.879746838461
Sat Dec 30 01:36:03 2017.csv:1,16,1,0,3,1,1,128,0,0,9,2,0,16,1,1,3,2,1,64,0,1,8,2,0,64,1,1,0,1,1,32,1,1,6,2,0,8,1,0,9,1,0,8,1,0,3,0,0,512,0,0,0,1,16,1,0,1,3,0.299078596921,0.898734181742
Sat Dec 30 01:36:03 2017.csv:1,16,0,0,3,2,0,64,1,0,10,0,1,128,0,0,2,2,1,64,0,1,8,2,1,64,1,1,0,1,0,16,1,0,1,2,1,64,0,1,9,2,0,32,1,0,1,0,0,512,0,0,0,1,16,1,1,1,3,0.286733426625,0.889240512365"""


def get_models():
    models = []
    gs = [list(map(int, l.split(":")[-1].split(",")[:-2])) for l in txt.splitlines()]
    gs2 = [list(map(int, l.split(":")[-1].split(",")[:-2])) for l in txt2.splitlines()]
    
    for g in gs:
        models.append(gh.decode(g))
    for g in gs2:
        models.append(gh.decode(g))
    return models
    
    
def get_model(img_shape=(75, 75, 2), num_classes=1, f=8, h=128):

    """
    This model structure is inspired and modified from the following kernel
    https://www.kaggle.com/knowledgegrappler/a-keras-prototype-0-21174-on-pl
    img_shape: dimension for input image
    f: filters of first conv blocks and generate filters in the following 
       blocks acorrdingly 
    h: units in dense hidden layer
    """ 
    
    #model
    bn_model = 0
    p_activation = 'elu'
    
    #
    input_img = Input(shape=img_shape, name='img_inputs')
    input_img_bn = BatchNormalization(momentum=bn_model)(input_img)
    #
    input_meta = Input(shape=[1], name='angle')
    input_meta_bn = BatchNormalization(momentum=bn_model)(input_meta)
    
    #img_1
    #img_1:block_1
    img_1 = conv_block(input_img_bn, nf=f, k=3, s=1, nb=3, p_act=p_activation)
    img_1 = bn_pooling(img_1, k=3, s=3, m=0)
    
    #img_1:block_2
    f*=2
    img_1 = Dropout(0.2)(img_1)
    img_1 = conv_block(img_1, nf=f, k=3, s=1, nb=3, p_act=p_activation)
    img_1 = bn_pooling(img_1, k=3, s=2, m=0)
    
    #img_1:block_3
    f*=2
    img_1 = Dropout(0.2)(img_1)
    img_1 = conv_block(img_1, nf=f, k=3, s=1, nb=3, p_act=p_activation)
    img_1 = bn_pooling(img_1, k=3, s=3, m=0)
    
    #img_1:block_4
    f*=2
    img_1 = Dropout(0.2)(img_1)
    img_1 = conv_block(img_1, nf=f, k=3, s=1, nb=3, p_act=p_activation)
    img_1 = Dropout(0.2)(img_1)
    img_1 = BatchNormalization(momentum=bn_model)(GlobalMaxPooling2D()(img_1))
    
    #img 2
    img_2 = conv_block(input_img_bn, nf=f, k=3, s=1, nb=6, p_act=p_activation)
    img_2 = Dropout(0.2)(img_2)
    img_2 = BatchNormalization(momentum=bn_model)(GlobalMaxPooling2D()(img_2))
    
    #full connect
    concat = (Concatenate()([img_1, img_2, input_meta_bn]))
    x = dense_block(concat, h=h)
    x = dense_block(x, h=h)
    output = Dense(num_classes, activation='sigmoid')(x)
    
    model = Model([input_img, input_meta],  output)

    model.summary()
    
    return model

if __name__ == '__main__':
    model = get_model()

