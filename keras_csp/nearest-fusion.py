    # -*- coding: utf-8 -*-


from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from keras.layers import *

from keras import backend as K
from .keras_layer_L2Normalization import L2Normalization
import numpy as np
import keras, math

def identity_block(input_tensor, kernel_size, filters, stage, block, dila=(1,1), trainable=True):

    nb_filter1, nb_filter2, nb_filter3 = filters
    
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, (1, 1), name=conv_name_base + '2a', trainable=trainable)(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, (kernel_size, kernel_size), dilation_rate=dila, padding='same', name=conv_name_base + '2b', trainable=trainable)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, (1, 1), name=conv_name_base + '2c', trainable=trainable)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), dila=(1,1), trainable=True):

    nb_filter1, nb_filter2, nb_filter3 = filters
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, (1, 1), strides=strides, name=conv_name_base + '2a', trainable=trainable)(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, (kernel_size, kernel_size), dilation_rate=dila, padding='same', name=conv_name_base + '2b', trainable=trainable)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, (1, 1), name=conv_name_base + '2c', trainable=trainable)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Convolution2D(nb_filter3, (1, 1), strides=strides, name=conv_name_base + '1', trainable=trainable)(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x


def nn_p3p4p5(img_input=None,offset=True, num_scale=1, trainable=False):
 
    bn_axis = 3
    x = ZeroPadding2D((3, 3))(img_input)
    x = Convolution2D(64, (7, 7), strides=(2, 2), name='conv1', trainable=False)(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), trainable=False)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', trainable=False)
    stage2 = identity_block(x, 3, [64, 64, 256], stage=2, block='c', trainable=False)
    # print('stage2: ', stage2._keras_shape[1:])
    x = conv_block(stage2, 3, [128, 128, 512], stage=3, block='a', trainable=trainable)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', trainable=trainable)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', trainable=trainable)
    stage3 = identity_block(x, 3, [128, 128, 512], stage=3, block='d', trainable=trainable)
    # print('stage3: ', stage3._keras_shape[1:])
    x = conv_block(stage3, 3, [256, 256, 1024], stage=4, block='a', trainable=trainable)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b', trainable=trainable)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c', trainable=trainable)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d', trainable=trainable)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e', trainable=trainable)
    stage4 = identity_block(x, 3, [256, 256, 1024], stage=4, block='f', trainable=trainable)
    # print('stage4: ', stage4._keras_shape[1:])
    x = conv_block(stage4, 3, [512, 512, 2048], stage=5, block='a', strides=(1, 1), dila=(2, 2),
                   trainable=trainable)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', dila=(2, 2), trainable=trainable)
    stage5 = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', dila=(2, 2), trainable=trainable)
    # print('stage5: ', stage5._keras_shape[1:])
    x = conv_block(stage5, 3, [512, 512, 4096], stage=6, block='a', strides=(1, 1), dila=(2, 2),
                   trainable=trainable)
    x = identity_block(x, 3, [512, 512, 4096], stage=6, block='b', dila=(2, 2), trainable=trainable)
    stage6 = identity_block(x, 3, [512, 512, 4096], stage=6, block='c', dila=(2, 2), trainable=trainable)
   
    
    C2 = stage2
    C3 = stage3
    C4 = stage4
    C5 = stage5
    C6 = stage6
   
    P6 = Conv2D(256, (1, 1), name='fpn_c6p6')(C6)
    P5 = Add(name="fpn_p5add")([P6,Conv2D(256, (1, 1), name='fpn_c5p5')(C5)])
    
    Pp5 = Deconvolution2D(256, kernel_size=4, strides=2, padding='same',kernel_initializer='glorot_normal', name='P4', trainable=trainable)(P5)
    P5_1= L2Normalization(gamma_init=10, name='P5norm_1')(Pp5)

    P4 = Deconvolution2D(256, kernel_size=4, strides=2, padding='same',kernel_initializer='glorot_normal', name='c4', trainable=trainable)(C4)
    P4_1= L2Normalization(gamma_init=10, name='P4norm_1')(P4)
    P3 = Add(name="P4C3")([P4_1,Conv2D(256, (1, 1), name='P4_1')(C3)])
    
    P3_5= Add(name="P3P5")([P5_1,Conv2D(256, (1, 1), name='P3')(P3)])
   
    P3_up = Deconvolution2D(256, kernel_size=4, strides=2, padding='same',
                            kernel_initializer='glorot_normal', name='P3up', trainable=trainable)(P3_5)
   
    feat = Convolution2D(256, (3, 3), padding='same', kernel_initializer='glorot_normal', name='feat',
                         trainable=trainable)(P3_up)
    feat = BatchNormalization(axis=bn_axis, name='bn_feat')(feat)
    feat = Activation('relu')(feat)

    x_class = Convolution2D(1, (1, 1), activation='sigmoid',
                            kernel_initializer='glorot_normal',
                            bias_initializer=prior_probability_onecls(probability=0.01),
                            name='center_cls', trainable=trainable)(feat)
    x_regr = Convolution2D(num_scale, (1, 1), activation='linear', kernel_initializer='glorot_normal',
                           name='height_regr', trainable=trainable)(feat)

    if offset:
        x_offset = Convolution2D(2, (1, 1), activation='linear',
kernel_initializer='glorot_normal',name='offset_regr', trainable=trainable)(feat)
        return [x_class, x_regr, x_offset]
    else:
        return [x_class, x_regr]

 # focal loss like
def prior_probability_onecls(num_class=1, probability=0.01):
	def f(shape, dtype=keras.backend.floatx()):
		assert(shape[0] % num_class == 0)
		# set bias to -log((1 - p)/p) for foregound
		result = np.ones(shape, dtype=dtype) * -math.log((1 - probability) / probability)
		# set bias to -log(p/(1 - p)) for background
		return result
	return f 
 

