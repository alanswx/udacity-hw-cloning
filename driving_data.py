import scipy.misc
import random
import cv2
import numpy as np
from sklearn.utils import shuffle


xs = []
ys = []

#points to the end of the last batch
train_batch_pointer = 0
val_batch_pointer = 0
steering_camera_offset = 0.15
#steering_camera_offset = 0.25

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
remove = 0
for line in csv_data:
  #print(line)
  # remove 90% of straight scenes
  i = i + 1
  if (abs(float(line[3])) < 1e-5):
    remove=remove+1
    if (remove==10):
      remove=0
      xs.append( path+'/'+line[0].decode('UTF-8').strip())
      ys.append(float(line[3]))
      # add the left image
      #Axs.append( path+'/'+line[1].decode('UTF-8').strip())
      #Ays.append(float(line[3])+steering_camera_offset)
      # add the right image
      #Axs.append( path+'/'+line[0].decode('UTF-8').strip())
      #Ays.append(float(line[3])-steering_camera_offset)
  else:
    #if (i % 10) :
    #  if (abs(float(line[3])) > 0.11):
    #    xs.append( "flip"+path+'/'+line[0].decode('UTF-8').strip())
    #    ys.append(float(line[3])*-1)

    xs.append( path+'/'+line[0].decode('UTF-8').strip())
    ys.append(float(line[3]))
    # add the left image
    #Axs.append( path+'/'+line[1].decode('UTF-8').strip())
    #Ays.append(float(line[3])+steering_camera_offset)
    # add the right image
    #Axs.append( path+'/'+line[0].decode('UTF-8').strip())
    #Ays.append(float(line[3])-steering_camera_offset)
    # add the left image
    #xs.append( path+'/'+line[1].decode('UTF-8').strip())
    #ys.append(float(line[3])+steering_camera_offset)
    # add the right image
    #xs.append( path+'/'+line[0].decode('UTF-8').strip())
    #ys.append(float(line[3])-steering_camera_offset)
    # more frames of the same?
    #xs.append( path+'/'+line[0].decode('UTF-8').strip())
    #ys.append(float(line[3]))
    # add the left image
    #xs.append( path+'/'+line[1].decode('UTF-8').strip())
    #ys.append(float(line[3])+steering_camera_offset)
    # add the right image
    #xs.append( path+'/'+line[0].decode('UTF-8').strip())
    #ys.append(float(line[3])-steering_camera_offset)
#  xs.append( path+'/'+line[0].decode('UTF-8').strip())
#  ys.append(float(line[3]))
#  xs.append( path+'/'+line[0].decode('UTF-8').strip())
#  ys.append(float(line[3]))
  # flip
#  xs.append( "flip"+path+'/'+line[0].decode('UTF-8').strip())
#  ys.append(float(line[3])*-1)
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


def get_dataset(func):
    images= [func(x) for x in train_xs]
    return images, train_ys

#
# fix this to either use a generator, or call through the proper functions
#
'''
def get_validation_dataset():
    images= [np.float32(cv2.resize(cv2.imread(x, 1), (200, 66))) / 255.0 for x in val_xs]
    return np.array(images), np.array(val_ys)
'''

def get_validation_dataset(func):
    #images= [np.float32(cv2.resize(cv2.imread(x, 1), (200, 66))) / 255.0 for x in val_xs]
    images= [func(x) for x in val_xs]
    return np.array(images), np.array(val_ys)

def process_image_comma_pixels(image):
    top_crop = 55
    bottom_crop = 135
    mean=0

    image = image[top_crop:bottom_crop, :, :]
   #image=cv2.copyMakeBorder(image, top=top_crop, bottom=(160-bottom_crop) , left=0, right=0, borderType= cv2.BORDER_CONSTANT, value=[mean,mean,mean] )

    #return np.array(image)[None, :, :, :].transpose(0, 3, 1, 2)
    (h, w) = image.shape[:2]
    #randomize brightness
    brightness = random.uniform (-0.3, 0.3)
    image = np.add(image, brightness)

    # black squares from Russian demo
    rect_w = 25
    rect_h = 25
    rect_count = 30
    for i in range (rect_count):
        pt1 = (random.randint (0, w), random.randint (0, h))
        pt2 = (pt1[0] + rect_w, pt1[1] + rect_h)
        cv2.rectangle(image, pt1, pt2, (-0.5, -0.5, -0.5), -1)

    #rotation and scaling
    rot = 1
    scale = 0.02
    Mrot = cv2.getRotationMatrix2D((h/2,w/2),random.uniform(-rot, rot), random.uniform(1.0 - scale, 1.0 + scale))

    #affine transform and shifts
    pts1 = np.float32([[0,0],[w,0],[w,h]])
    a = 0
    shift = 2
    shiftx = random.randint (-shift, shift);
    shifty = random.randint (-shift, shift);
    pts2 = np.float32([[
                0 + random.randint (-a, a) + shiftx,
                0 + random.randint (-a, a) + shifty
            ],[
                w + random.randint (-a, a) + shiftx,
                0 + random.randint (-a, a) + shifty
            ],[
                w + random.randint (-a, a) + shiftx,
                h + random.randint (-a, a) + shifty
            ]])
    M = cv2.getAffineTransform(pts1,pts2)

    image = cv2.warpAffine(
            cv2.warpAffine (
                image
                , Mrot, (w, h)
            )
            , M, (w,h)
        )



    image = cv2.resize(image, (320, 160) ) / 255.0 
    return np.array(image)[None, :, :, :].transpose(0, 3, 1, 2)

def process_image_comma(name):
   if 'flip' == name[0:4]:
      name = name[4:]
      image = cv2.imread(name)
      image = cv2.flip(image, 1)
   else: 
      image = cv2.imread(name)
   return process_image_comma_pixels(image)[0]


def process_image_sully_pixels(image):
    top_crop = 55
    bottom_crop = 135
    mean=0

    image= image[top_crop:bottom_crop, :, :]

    (h, w) = image.shape[:2]
    #randomize brightness
    brightness = random.uniform (-0.3, 0.3)
    image = np.add(image, brightness)

    # black squares from Russian demo
    rect_w = 25
    rect_h = 25
    rect_count = 30
    for i in range (rect_count):
        pt1 = (random.randint (0, w), random.randint (0, h))
        pt2 = (pt1[0] + rect_w, pt1[1] + rect_h)
        cv2.rectangle(image, pt1, pt2, (-0.5, -0.5, -0.5), -1)

   #pixels=cv2.copyMakeBorder(image, top=top_crop, bottom=(160-bottom_crop) , left=0, right=0, borderType= cv2.BORDER_CONSTANT, value=[mean,mean,mean] )
    #rotation and scaling
    rot = 1
    scale = 0.02
    Mrot = cv2.getRotationMatrix2D((h/2,w/2),random.uniform(-rot, rot), random.uniform(1.0 - scale, 1.0 + scale))

    #affine transform and shifts
    pts1 = np.float32([[0,0],[w,0],[w,h]])
    a = 0
    shift = 2
    shiftx = random.randint (-shift, shift);
    shifty = random.randint (-shift, shift);
    pts2 = np.float32([[
                0 + random.randint (-a, a) + shiftx,
                0 + random.randint (-a, a) + shifty
            ],[
                w + random.randint (-a, a) + shiftx,
                0 + random.randint (-a, a) + shifty
            ],[
                w + random.randint (-a, a) + shiftx,
                h + random.randint (-a, a) + shifty
            ]])
    M = cv2.getAffineTransform(pts1,pts2)

    image = cv2.warpAffine(
            cv2.warpAffine (
                image
                , Mrot, (w, h)
            )
            , M, (w,h)
        )

    return np.float32(cv2.resize(image, (200, 66) )) / 255.0 

def process_image_gray_pixels(image):
    image = np.copy (image)

    top_crop = 55
    bottom_crop = 135
    mean=0

    #pixels = image[top_crop:bottom_crop, :]

    (h, w) = image.shape[:2]
    #randomize brightness
    brightness = random.uniform (-0.3, 0.3)
    image = np.add(image, brightness)

    # black squares from Russian demo
    rect_w = 25
    rect_h = 25
    rect_count = 30
    for i in range (rect_count):
        pt1 = (random.randint (0, w), random.randint (0, h))
        pt2 = (pt1[0] + rect_w, pt1[1] + rect_h)
        cv2.rectangle(image, pt1, pt2, (-0.5, -0.5, -0.5), -1)

   #pixels=cv2.copyMakeBorder(image, top=top_crop, bottom=(160-bottom_crop) , left=0, right=0, borderType= cv2.BORDER_CONSTANT, value=[mean,mean,mean] )
    #rotation and scaling
    rot = 1
    scale = 0.02
    Mrot = cv2.getRotationMatrix2D((h/2,w/2),random.uniform(-rot, rot), random.uniform(1.0 - scale, 1.0 + scale))

    #affine transform and shifts
    pts1 = np.float32([[0,0],[w,0],[w,h]])
    a = 0
    shift = 2
    shiftx = random.randint (-shift, shift);
    shifty = random.randint (-shift, shift);
    pts2 = np.float32([[
                0 + random.randint (-a, a) + shiftx,
                0 + random.randint (-a, a) + shifty
            ],[
                w + random.randint (-a, a) + shiftx,
                0 + random.randint (-a, a) + shifty
            ],[
                w + random.randint (-a, a) + shiftx,
                h + random.randint (-a, a) + shifty
            ]])
    M = cv2.getAffineTransform(pts1,pts2)

    image = cv2.warpAffine(
            cv2.warpAffine (
                image
                , Mrot, (w, h)
            )
            , M, (w,h)
        )

    #image = cv2.resize(image, (320, 160) ) / 255.0 
    return image

def open_image_gray(name):
   if 'flip' == name[0:4]:
      name = name[4:]
      image = cv2.imread(name,cv2.IMREAD_GRAYSCALE)
      image = cv2.flip(image, 1)
   else: 
      image = cv2.imread(name,cv2.IMREAD_GRAYSCALE)
   image = np.subtract(np.divide(np.array(image).astype(np.float32), 255.0), 0.5)
   return image

def process_image_gray(name):
   #print(name)
   image = open_image_gray(name)
   return process_image_gray_pixels(image)



def process_image_sully(name):
   if 'flip' == name[0:4]:
      name = name[4:]
      image = cv2.imread(name)
      image = cv2.flip(image, 1)
   else: 
      image = cv2.imread(name)
   #print(name)
   return process_image_sully_pixels(image)
 
def comma_y_func(y):
   return y * 180 / scipy.pi
   #return y 
def russia_y_func(y):
   return y
def sully_y_func(y):
   # russian - add noise to steering angle
   y= y+ np.random.normal (0, 0.005)
   return y 
   
def generator(X_items,y_items,batch_size,x_func=process_image_sully,y_func=sully_y_func):
  #print("inside generator")
  gen_state = 0
  bs = batch_size
  while 1:
    if gen_state > len(X_items):
      bs = batch_size
      gen_state = 0
      # reshuffle batch
      X_items, y_items= shuffle(X_items, y_items)
      
    if gen_state + batch_size > len(X_items):
      bs = len(X_items) - gen_state
      #gen_state=0
    paths = X_items[gen_state : gen_state + bs]
    yb = y_items[gen_state : gen_state + bs]
    y = [ y_func(y1) for y1 in  yb]
    X =  [x_func(x)  for x in paths]
    gen_state = gen_state + batch_size 
    yield np.asarray(X), np.asarray(y)

def generator_2(images_arr,steering_arr,batch_size,x_func=process_image_sully,y_func=sully_y_func):
    print("AJS HERE")
    last_index = len (images_arr) - 1
    while 1:
        batch_img = []
        batch_steering = []
        for i in range (batch_size):

            idx_img = random.randint (0, last_index)
            im = x_func(images_arr[idx_img])
            steering = y_func(steering_arr[idx_img])
            if (random.uniform (0, 1) > 0.5):
                im = cv2.flip (im, 1)
                steering = - steering
            steering = steering + np.random.normal (0, 0.005)

            #im, steering = augment_record (images_arr [idx_img], steering_arr[idx_img])
            #im, steering = images_arr [idx_img], steering_arr[idx_img]

            batch_img.append (im)
            batch_steering.append (steering)

        batch_img = np.asarray (batch_img)
        batch_steering = np.asarray (batch_steering)
        yield (batch_img, batch_steering)




'''
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
'''
