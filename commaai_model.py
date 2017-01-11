#!/usr/bin/env python
"""
Steering angle prediction model
"""
import os
import argparse
import json
import pickle
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D
from sklearn.model_selection import train_test_split
import cv2

#
# open and look at the data we are going to use to train
#

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import scipy.misc


steering_camera_offset = 0.15


path = '/data1/udacity/simulator/data'
img_path = path +'/IMG'
csv_file = path +'/driving_log.csv'

# open the CSV file and loop through each line
# load the CSV so we can have labels
csv_data=np.recfromcsv(csv_file, delimiter=',', filling_values=np.nan, case_sensitive=True, deletechars='', replace_space=' ')
print(csv_data.shape)
#print(csv_data)

# center, left, right, steering angle, throttle, break, speed

# preprocess the data

X_full_name = []
y_full_angle= []

for line in csv_data:
  #print(line)
  X_full_name.append( path+'/'+line[0].decode('UTF-8').strip())
  y_full_angle.append(float(line[3]))
  # flip
  #X_full_name.append( "flip"+path+'/'+line[0].decode('UTF-8').strip())
  #y_full_angle.append(float(line[3])*-1)
  # add the left image
  X_full_name.append( path+'/'+line[1].decode('UTF-8').strip())
  y_full_angle.append(float(line[3])+steering_camera_offset)
  # add the right image
  X_full_name.append( path+'/'+line[0].decode('UTF-8').strip())
  y_full_angle.append(float(line[3])-steering_camera_offset)
  #print(X_full_name)
 

#
#  shuffle the data and split it into a validation set
# 

#X_train, X_validation, y_train, y_validation = train_test_split(X_full_name, y_full_angle, test_size=0.20, random_state=42)
X_train, X_validation, y_train, y_validation = train_test_split(X_full_name, y_full_angle, test_size=0.01, random_state=42)


print(len(X_full_name)) 


def process_image(name):
   top_crop = 55
   bottom_crop = 135
   mean=0
   if 'flip' == name[0:4]:
      name = name[4:]
      image = cv2.imread(name)
      image = cv2.flip(image, 1)
   else: 
      image = cv2.imread(name)

   image = image[top_crop:bottom_crop, :, :]
   image=cv2.copyMakeBorder(image, top=top_crop, bottom=(160-bottom_crop) , left=0, right=0, borderType= cv2.BORDER_CONSTANT, value=[mean,mean,mean] )

   return np.array(image)[None, :, :, :].transpose(0, 3, 1, 2)[0]
 

def generator(X_items,y_items,batch_size):
  #print("inside generator")
  gen_state = 0
  bs = batch_size
  while 1:
    if gen_state > len(X_items):
      bs = batch_size
      gen_state = 0
    if gen_state + batch_size > len(X_items):
      bs = len(X_items) - gen_state
      #gen_state=0
    paths = X_items[gen_state : gen_state + bs]
    yb = y_items[gen_state : gen_state + bs]
    #y = [ (y1 * 2 ) for y1 in  yb]
    y = [ (y1 * 180 / scipy.pi) for y1 in  yb]
    X =  [process_image(x)  for x in paths]
    #X =  [np.float32((cv2.imread(x, 1)[None, :, :, :].transpose(0, 3, 1, 2) )) / 255.0 for x in paths]
    gen_state = gen_state + batch_size 
    #print(len(X))
    #print(X[0][0].shape)
    #print(np.asarray(X).shape)
    #print(np.asarray(X[0]).shape)
    yield np.asarray(X), np.asarray(y)

def get_model(time_len=1):
  ch, row, col = 3, 160, 320  # camera format

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

  #model.load_weights("./outputs/steering_model/steering_angle.keras" )

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

  model = get_model()
  res=model.fit_generator(
    generator(X_train,y_train,args.batch),
    samples_per_epoch=len(X_train),
    nb_epoch=args.epoch,
    validation_data=generator(X_validation,y_validation,args.batch),
    nb_val_samples=len(X_validation)
  )
  print("Saving model weights and configuration file.")

  if not os.path.exists("./outputs/steering_model"):
      os.makedirs("./outputs/steering_model")

  model.save_weights("./outputs/steering_model/steering_angle.keras", True)
  with open('./outputs/steering_model/steering_angle.json', 'w') as outfile:
    json.dump(model.to_json(), outfile)


  history=res.history
  with open('history.p','wb') as f:
     pickle.dump(history,f)

