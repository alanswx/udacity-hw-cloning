import scipy.misc
import random
import cv2
import numpy as np
xs = []
ys = []

#points to the end of the last batch
train_batch_pointer = 0
val_batch_pointer = 0
steering_camera_offset = 0.15

#read data.txt
#with open("driving_dataset/data.txt") as f:
#    for line in f:
#        xs.append("driving_dataset/" + line.split()[0])
        #the paper by Nvidia uses the inverse of the turning radius,
        #but steering wheel angle is proportional to the inverse of turning radius
        #so the steering wheel angle in radians is used as the output
#        ys.append(float(line.split()[1]) * scipy.pi / 180)
        
path = '/data1/udacity/simulator/data'
img_path = path +'/IMG'
csv_file = path +'/driving_log.csv'

# open the CSV file and loop through each line
# load the CSV so we can have labels
csv_data=np.recfromcsv(csv_file, delimiter=',', filling_values=np.nan, case_sensitive=True, deletechars='', replace_space=' ')
i  = 0
for line in csv_data:
  #print(line)
  xs.append( path+'/'+line[0].decode('UTF-8').strip())
  ys.append(float(line[3]))
  # flip
  #xs.append( "flip"+path+'/'+line[0].decode('UTF-8').strip())
  #ys.append(float(line[3])*-1)
  # add the left image
  xs.append( path+'/'+line[1].decode('UTF-8').strip())
  ys.append(float(line[3])+steering_camera_offset)
  # add the right image
  xs.append( path+'/'+line[0].decode('UTF-8').strip())
  ys.append(float(line[3])-steering_camera_offset)
  #print(X_full_name)

#get number of images
num_images = len(xs)

#shuffle list of images
c = list(zip(xs, ys))
random.shuffle(c)
xs, ys = zip(*c)

train_xs = xs[:int(len(xs) * 0.8)]
train_ys = ys[:int(len(xs) * 0.8)]

val_xs = xs[-int(len(xs) * 0.2):]
val_ys = ys[-int(len(xs) * 0.2):]

num_train_images = len(train_xs)
num_val_images = len(val_xs)


def get_dataset():
    images= [np.float32(cv2.resize(cv2.imread(x, 1), (200, 66))) / 255.0 for x in train_xs]
    return images, train_ys

def get_validation_dataset():
    images= [np.float32(cv2.resize(cv2.imread(x, 1), (200, 66))) / 255.0 for x in val_xs]
    return np.array(images), np.array(val_ys)

def process_image_comma_pixels(image):
   top_crop = 55
   bottom_crop = 135
   mean=0

   image = image[top_crop:bottom_crop, :, :]
   image=cv2.copyMakeBorder(image, top=top_crop, bottom=(160-bottom_crop) , left=0, right=0, borderType= cv2.BORDER_CONSTANT, value=[mean,mean,mean] )

   return np.array(image)[None, :, :, :].transpose(0, 3, 1, 2)[0]

def process_image_comma(name):
   if 'flip' == name[0:4]:
      name = name[4:]
      image = cv2.imread(name)
      image = cv2.flip(image, 1)
   else: 
      image = cv2.imread(name)
   return process_image_comma_pixels(image)


def process_image_sully_pixels(pixels):
   return np.float32(cv2.resize(pixels, (200, 66) )) / 255.0 

def process_image_sully(name):
   return process_image_sully_pixels(cv2.imread(name, 1))
 
def comma_y_func(y):
   return y * 180 / scipy.pi
def sully_y_func(y):
   return y 
   
def generator(X_items,y_items,batch_size,x_func=process_image_sully,y_func=sully_y_func):
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
    y = [ y_func(y1) for y1 in  yb]
    X =  [x_func(x)  for x in paths]
    gen_state = gen_state + batch_size 
    yield np.asarray(X), np.asarray(y)





def generate_arrays_from_file(path = "driving_dataset/data.txt"):
    gen_state = 0
    print ("Got lines!")
    while 1:
        if gen_state + 100 > len(train_xs):
            gen_state = 0
        paths = train_xs[gen_state : gen_state + 100]
        y = train_ys[gen_state : gen_state + 100]
        X =  [np.float32(cv2.resize(cv2.imread(x, 1), (200, 66) )) / 255.0 for x in paths]
        gen_state = gen_state + 100
        #print(X[0].shape)
        yield np.array(X), np.array(y)
