import argparse
import base64
import json

import cv2

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO
import scipy.misc

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
from keras.optimizers import Adam

import driving_data
# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf


sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = np.asarray(image)
    print(image_array.shape)
    image_array = cv2.cvtColor (image_array, cv2.COLOR_BGR2GRAY)
    image_array = np.subtract(np.divide(np.array(image_array).astype(np.float32), 255.0), 0.5)
    transformed_image_array = image_array[None, :, :]

    #transformed_image_array=driving_data.process_image_gray_pixels(image_array)
    print(transformed_image_array.shape)
    ''' 
    image_array = image_array[55:135, :, :]
    mean=0
    image_array=cv2.copyMakeBorder(image_array, top=55, bottom=25 , left=0, right=0, borderType= cv2.BORDER_CONSTANT, value=[mean,mean,mean] )
    #cv2.resize(image, (160,320))

    b = image_array[None, :, :, :].transpose(0, 3, 1, 2)
    print(b.shape)

    transformed_image_array = b
    '''    
    #transformed_image_array = image_array[None, :, :, :]
    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    print("about to call predict")
    steering_angle = float(model.predict(transformed_image_array, batch_size=1))
    #steering_angle =  steering_angle * scipy.pi / 180

    print("after predict")
    #steering_angle =  steering_angle * scipy.pi / 180 
    #steering_angle =  steering_angle * scipy.pi / 180 
   # steering_angle =  steering_angle / 2
    print("steering angle"+str(steering_angle))
    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    speed = float(speed)
  # TODO - change this
    if speed < 10.0:
        throttle = 0.7
    elif speed < 15.0:
        throttle = 0.4
    elif speed < 22.0:
        throttle = 0.18
    else:
        throttle = 0.15

    #throttle = 0.2
    print(steering_angle, throttle)
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        # NOTE: if you saved the file by calling json.dump(model.to_json(), ...)
        # then you will have to call:
        #
        model = model_from_json(json.loads(jfile.read()))
        model.summary()
        #
        # instead.
        #model = model_from_json(jfile.read())


    learning_rate=0.0001
    model.compile(Adam(lr=learning_rate), "mse")
    weights_file = args.model.replace('json', 'keras')
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
