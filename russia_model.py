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

from sklearn.model_selection import train_test_split
import cv2

import driving_data

#
# open and look at the data we are going to use to train
#

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import scipy.misc

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



def get_model2(time_len=1):
  ch, row, col = 3, 160, 320  # camera format
  #ch, row, col = 3, 80, 160  # camera format

  model = Sequential()
  model.add(Lambda(lambda x: x/127.5 - 1.,
            input_shape=(ch, row, col),
            output_shape=(ch, row, col)))
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
  model.add(ELU())
  model.add(Dense(1))

  #model.load_weights("./outputs/russia_steering_model/steering_angle.keras" )

  #learning_rate=0.00075
  #learning_rate=0.001
  learning_rate=0.0015
  #learning_rate=0.003
  model.compile(optimizer=Adam(lr=learning_rate), loss="mse")

  return model


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Steering angle model trainer')
  parser.add_argument('--host', type=str, default="localhost", help='Data server ip address.')
  parser.add_argument('--port', type=int, default=5557, help='Port of server.')
  parser.add_argument('--val_port', type=int, default=5556, help='Port of server for validation dataset.')
  parser.add_argument('--batch', type=int, default=64, help='Batch size.')
  parser.add_argument('--epoch', type=int, default=200, help='Number of epochs.')
  parser.add_argument('--epochsize', type=int, default=10000, help='How many frames per epoch.')
  parser.add_argument('--skipvalidate', dest='skipvalidate', action='store_true', help='Multiple path output.')
  parser.set_defaults(skipvalidate=False)
  parser.set_defaults(loadweights=False)
  args = parser.parse_args()

 
  print(args.batch)
  checkpoint_path="weights.{epoch:02d}-{val_loss:.2f}.hdf5"
  checkpoint = ModelCheckpoint(checkpoint_path, verbose=1, save_best_only=True, save_weights_only=False, mode='auto')
  earlyStop =  EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=1, mode='auto')

  model = get_model()
  res=model.fit_generator(
    driving_data.generator(driving_data.train_xs,driving_data.train_ys,args.batch,driving_data.process_image_gray,driving_data.comma_y_func),
    samples_per_epoch=len(driving_data.train_xs),
    nb_epoch=args.epoch,
    validation_data=driving_data.generator(driving_data.val_xs,driving_data.val_ys,args.batch,driving_data.process_image_gray,driving_data.comma_y_func),
    nb_val_samples=len(driving_data.val_xs), callbacks = [ checkpoint]
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

