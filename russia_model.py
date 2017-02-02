#!/usr/bin/env python
"""
Steering angle prediction model
"""
import os
import argparse
import json
import pickle
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D
from keras.layers import Dense, Activation, Reshape, Merge
from keras.layers.core import Flatten, Reshape, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D, AveragePooling1D
from keras.optimizers import Adam, SGD
from keras.utils.np_utils import to_categorical
from keras.callbacks import Callback as KerasCallback
from keras.layers import Input, LSTM, Dense, merge
from keras.models import Model
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, SimpleRNN, Reshape, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.regularizers import l2
from keras.models import Sequential
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D,ELU
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras import backend as K
from keras.optimizers import SGD
from keras.layers import Dense, Dropout, Activation, Flatten
import tensorflow as tf
from keras.layers import Input, LSTM, Dense, merge
from keras.models import Model
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, SimpleRNN, Reshape, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.regularizers import l2




from sklearn.model_selection import train_test_split
import cv2

import driving_data

#
# open and look at the data we are going to use to train
#

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import scipy.misc

def nvidia_net():
    model = Sequential()
    #lambda?
    p=0.33
    model.add(Reshape ((160, 320, 1), input_shape=(160, 320)))
    model.add(Convolution2D(24, 5, 5, init = 'he_normal', subsample= (2, 2), name='conv1_1', border_mode='valid',input_shape=(160, 320, 1)))
    model.add(ELU())
    model.add(Convolution2D(36, 5, 5, init = 'he_normal', subsample= (2, 2), border_mode='valid',name='conv2_1'))
    model.add(ELU())
    model.add(Convolution2D(48, 5, 5, init = 'he_normal', subsample= (2, 2), border_mode='valid',name='conv3_1'))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, init = 'he_normal', subsample= (1, 1), border_mode='valid',name='conv4_1'))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, init = 'he_normal', subsample= (1, 1), border_mode='valid',name='conv4_2'))
    model.add(ELU())
    model.add(Flatten())
    model.add(Dense(1164, init = 'he_normal', name = "dense_0"))
    model.add(ELU())
    model.add(Dropout(p))
    model.add(Dense(100, init = 'he_normal',  name = "dense_1"))
    model.add(ELU())
    model.add(Dropout(p))
    model.add(Dense(50, init = 'he_normal', name = "dense_2"))
    model.add(ELU())
    model.add(Dropout(p))
    model.add(Dense(10, init = 'he_normal', name = "dense_3"))
    model.add(ELU())
    model.add(Dense(1, init = 'he_normal', name = "dense_4"))
    model.compile(loss = 'mse', optimizer = 'Adam')

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
    model.compile(loss = 'mse', optimizer = 'Adam')

    return model


def get_model():
  model = Sequential ([
        Reshape ((160, 320, 1), input_shape=(160, 320)),

        Convolution2D (24, 8, 8, border_mode='valid'),
        MaxPooling2D (pool_size=(2, 2)),
        Dropout (0.5),
        Activation ('relu'),

        #77x157
        Convolution2D (36, 5, 5, border_mode='valid'),
        MaxPooling2D (pool_size=(2, 2)),
        Dropout (0.5),
        Activation ('relu'),
        #37x77
        Convolution2D (48, 5, 5, border_mode='valid'),
        MaxPooling2D (pool_size=(2, 2)),
        Dropout (0.5),
        Activation ('relu'),

        #17x37
        Convolution2D (64, 3, 3, border_mode='valid'),
        MaxPooling2D (pool_size=(2, 2)),
        Dropout (0.5),
        Activation ('relu'),

        #8x18
        Convolution2D (64, 2, 2, border_mode='valid'),
        MaxPooling2D (pool_size=(2, 2)),
        Dropout (0.5),
        Activation ('relu'),

        #4x9
        Flatten (),

        Dense (1024),
        Dropout (0.5),
        Activation ('relu'),

        Dense (512),
        Dropout (0.5),
        Activation ('relu'),

        Dense (256),
        Activation ('relu'),

        Dense (128),
        Activation ('relu'),

        Dense (32),
        Activation ('tanh'),

        Dense (1)
  ])
  optimizer = Adam (lr=1e-4)

  model.compile (
    optimizer=optimizer,
    loss='mse',
    metrics=[]
  )

  return model

def vision_2D(dropout_frac=.2):
    '''
    Network with 4 convolutions, 2 residual shortcuts to predict angle.
    '''
    img_in = Input(shape=(160, 320), name='img_in')
    #img_in = Reshape ((160, 320, 1), input_shape=(160, 320), name='img_in')

    net =  Convolution2D(64, 6, 6, subsample=(4,4), name='conv0',input_shape=(160,320))(img_in)
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



def get_model2(time_len=1):
  ch, row, col = 3, 160, 320  # camera format
  #ch, row, col = 3, 80, 160  # camera format

  model = Sequential()
  model.add (Reshape ((160, 320, 1), input_shape=(160, 320)))
  #model.add(Lambda(lambda x: x/127.5 - 1.,
  #          input_shape=(ch, row, col),
  #          output_shape=(ch, row, col)))
  model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
  model.add(ELU())
  model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
  model.add(ELU())
  model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
  model.add(Flatten())
  model.add(Dropout(.2))
  model.add(ELU())
  model.add(Dense(512))
  model.add(Dropout(.5))
  model.add(Activation('relu'))
  model.add(Dense(32))
  model.add(Activation('tanh'))
  model.add(Dense(1))

  #model.load_weights("./outputs/russia_steering_model/steering_angle.keras" )

  #learning_rate=0.00075
  #learning_rate=0.001
  #learning_rate=0.0015
  learning_rate=1e-4
  #learning_rate=0.003
  model.compile(optimizer=Adam(lr=learning_rate), loss="mse")

  return model


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Steering angle model trainer')
  parser.add_argument('--host', type=str, default="localhost", help='Data server ip address.')
  parser.add_argument('--port', type=int, default=5557, help='Port of server.')
  parser.add_argument('--val_port', type=int, default=5556, help='Port of server for validation dataset.')
  parser.add_argument('--batch', type=int, default=112, help='Batch size.')
  parser.add_argument('--epoch', type=int, default=20, help='Number of epochs.')
  parser.add_argument('--epochsize', type=int, default=44800, help='How many frames per epoch.')
  parser.add_argument('--skipvalidate', dest='skipvalidate', action='store_true', help='Multiple path output.')
  parser.set_defaults(skipvalidate=False)
  parser.set_defaults(loadweights=False)
  args = parser.parse_args()

 
  print(args.batch)
  class SaveModel(KerasCallback):
    def on_epoch_end(self, epoch, logs={}):
        epoch += 1
        if (epoch>4):
            with open ('rmodel-' + str(epoch) + '.json', 'w') as file:
                file.write (model.to_json ())
                file.close ()

            model.save_weights ('rmodel-' + str(epoch) + '.h5')

  checkpoint_path="weights.{epoch:02d}-{val_loss:.2f}.hdf5"
  checkpoint = ModelCheckpoint(checkpoint_path, verbose=1, save_best_only=False, save_weights_only=False, mode='auto')
  earlyStop =  EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=1, mode='auto')

  model = get_model()
  #model = nvidia_net()
  model.summary()
  res=model.fit_generator(
    driving_data.generator(driving_data.train_xs,driving_data.train_ys,args.batch,driving_data.process_image_gray,driving_data.russia_y_func),
    samples_per_epoch=args.epochsize,
    nb_epoch=args.epoch,
    validation_data=driving_data.generator(driving_data.val_xs,driving_data.val_ys,args.batch,driving_data.open_image_gray,driving_data.russia_y_func),
    nb_val_samples=len(driving_data.val_xs), callbacks = [ SaveModel()]
  )
  print("Saving model weights and configuration file.")

  if not os.path.exists("./outputs/russia_steering_model"):
      os.makedirs("./outputs/russia_steering_model")

  model.save_weights("./outputs/russia_steering_model/steering_angle.keras", True)
  with open('./outputs/russia_steering_model/steering_angle.json', 'w') as outfile:
    json.dump(model.to_json(), outfile)


  history=res.history
  with open('history.p','wb') as f:
     pickle.dump(history,f)

