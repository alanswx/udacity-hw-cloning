from keras.models import Sequential
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D,ELU
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras import backend as K
from keras.optimizers import SGD
from keras.layers import Dense, Dropout, Activation, Flatten
import tensorflow as tf

def global_average_pooling(x):
    return tf.reduce_mean(x, (1, 2))

def global_average_pooling_shape(input_shape):
    return (input_shape[0], input_shape[3])

def atan_layer(x):
    return tf.mul(tf.atan(x), 2)

def atan_layer_shape(input_shape):
    return input_shape

def normal_init2(shape, name=None):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return K.variable(initial)

normal_init = 'normal'

from keras.layers import Input, LSTM, Dense, merge
from keras.models import Model
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, SimpleRNN, Reshape, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.regularizers import l2

def vision_2D(dropout_frac=.2):
    '''
    Network with 4 convolutions, 2 residual shortcuts to predict angle.
    '''
    #img_in = Input(shape=(120, 160, 3), name='img_in')
    img_in = Input(shape=(66, 200, 3), name='img_in')

    net =  Convolution2D(64, 6, 6, subsample=(4,4), name='conv0')(img_in)
    net =  Dropout(dropout_frac)(net)

    net =  Convolution2D(64, 3, 3, subsample=(2,2), name='conv1')(net)
    net =  Dropout(dropout_frac)(net)

    #Create residual to shortcut
    aux1 = Flatten(name='aux1_flat')(net)
    aux1 = Dense(64, name='aux1_dense')(aux1)

    net =  Convolution2D(128, 3, 3, subsample=(2,2), border_mode='same', name='conv2')(net)
    net =  Dropout(dropout_frac)(net)

    net =  Convolution2D(128, 3, 3, subsample=(2,2), border_mode='same', name='conv3')(net)
    net =  Dropout(dropout_frac)(net)

    aux2 = Flatten(name='aux2_flat')(net)
    aux2 = Dense(64, name='aux2_dense')(aux2)

    net = Flatten(name='net_flat')(net)
    net = Dense(512, activation='relu', name='net_dense1')(net)
    net =  Dropout(dropout_frac)(net)
    net = Dense(256, activation='relu', name='net_dense2')(net)
    net =  Dropout(dropout_frac)(net)
    net = Dense(128, activation='relu', name='net_dense3')(net)
    net =  Dropout(dropout_frac)(net)
    net = Dense(64, activation='linear', name='net_dense4')(net)

    net = merge([net, aux1, aux2], mode='sum') #combine residual layers
    angle_out = Dense(1, name='angle_out')(net)
    model = Model(input=[img_in], output=[angle_out])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def nvidia_net():
    model = Sequential()
    #lambda?
    p=0.33
    model.add(Convolution2D(24, 5, 5, init = 'normal', subsample= (2, 2), name='conv1_1', border_mode='valid',input_shape=(66, 200, 3)))
    model.add(Activation('relu'))
    model.add(Convolution2D(36, 5, 5, init = 'normal', subsample= (2, 2), border_mode='valid',name='conv2_1'))
    model.add(Activation('relu'))
    model.add(Convolution2D(48, 5, 5, init = 'normal', subsample= (2, 2), border_mode='valid',name='conv3_1'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, init = 'normal', subsample= (1, 1), border_mode='valid',name='conv4_1'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, init = 'normal', subsample= (1, 1), border_mode='valid',name='conv4_2'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(1164, init = 'normal', name = "dense_0"))
    model.add(Activation('relu'))
    model.add(Dropout(p))
    model.add(Dense(100, init = 'normal',  name = "dense_1"))
    model.add(Activation('relu'))
    model.add(Dropout(p))
    model.add(Dense(50, init = 'normal', name = "dense_2"))
    model.add(Activation('relu'))
    model.add(Dropout(p))
    model.add(Dense(10, init = 'normal', name = "dense_3"))
    model.add(Activation('tanh'))
    model.add(Dense(1, init = 'normal', name = "dense_4"))

    return model
    

def steering_net():
    #p=0.50
    p=0.33
    #p=0.25
    model = Sequential()
    model.add(Convolution2D(24, 5, 5, init = normal_init, subsample= (2, 2), name='conv1_1', input_shape=(66, 200, 3)))
    model.add(Activation('relu'))
    model.add(Convolution2D(36, 5, 5, init = normal_init, subsample= (2, 2), name='conv2_1'))
    model.add(Activation('relu'))
    model.add(Convolution2D(48, 5, 5, init = normal_init, subsample= (2, 2), name='conv3_1'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, init = normal_init, subsample= (1, 1), name='conv4_1'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, init = normal_init, subsample= (1, 1), name='conv4_2'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(1164, init = normal_init, name = "dense_0"))
    model.add(Activation('relu'))
    model.add(Dropout(p))
    model.add(Dense(100, init = normal_init,  name = "dense_1"))
    model.add(Activation('relu'))
    model.add(Dropout(p))
    model.add(Dense(50, init = normal_init, name = "dense_2"))
    model.add(Activation('relu'))
    model.add(Dropout(p))
    model.add(Dense(10, init = normal_init, name = "dense_3"))
    model.add(Activation('relu'))
    model.add(Dense(1, init = normal_init, name = "dense_4"))
    model.add(Lambda(atan_layer, output_shape = atan_layer_shape, name = "atan_0"))

    return model

def get_model():
    #model = steering_net()
    model = nvidia_net()
    model.compile(loss = 'mse', optimizer = 'Adam')
    return model

def load_model(path):
    #model = steering_net()
    model = nvidia_net()
    #model = vision_2D()
    model.load_weights(path)
    model.compile(loss = 'mse', optimizer = 'Adam')
    return model
