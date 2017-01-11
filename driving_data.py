import scipy.misc
import random
import cv2
import numpy as np
xs = []
ys = []

#points to the end of the last batch
train_batch_pointer = 0
val_batch_pointer = 0

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
#  xs.append(path+'/'+line[0].decode('UTF-8'))
#  ys.append(float(line[3]))
 # if (i%2 and abs(float(line[3]))<0.001):
 #    print("skip "+ str(float(line[3])))
 # else: 
     xs.append( path+'/'+line[0].decode('UTF-8').strip())
     ys.append(float(line[3]))
     #    add the left image
     xs.append( path+'/'+line[1].decode('UTF-8').strip())
     ys.append((float(line[3])+0.15))
     # add the right image
     xs.append( path+'/'+line[0].decode('UTF-8').strip())
     ys.append((float(line[3])-0.15))
  #i=i+1

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

def generate_arrays_from_file(path = "driving_dataset/data.txt"):
    import cv2
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
