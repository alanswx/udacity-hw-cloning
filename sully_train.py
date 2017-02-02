from keras.models import *
from keras.callbacks import *
import keras.backend as K
from sully_model import *
import cv2
import argparse
import driving_data
import pickle



#epochs=150
#epochs=20
#epochs=5
epochs=20
#epochs=12
def train():
        class SaveModel(Callback):
           def on_epoch_end(self, epoch, logs={}):
             epoch += 1
             if (epoch>0):
                 with open ('smodel-' + str(epoch) + '.json', 'w') as file:
                     file.write (model.to_json ())
                     file.close ()

                 model.save_weights ('smodel-' + str(epoch) + '.h5')

        #model = vision_2D()
        model = get_model()
        from keras.utils.visualize_util import plot
        plot(model, to_file='model.png',show_shapes=True)
        weights_file="./outputs/sully_steering_model/steering_angle.h5"
        #model = load_model(weights_file)

        print ("Loaded model")
        X, y = driving_data.get_validation_dataset(driving_data.process_image_sully)
        print (model.summary())
        print ("Loaded validation datasetset")
        print ("Total of", len(y) * 4)
        print ("Training..")
        checkpoint_path="weights.{epoch:02d}-{val_loss:.2f}.hdf5"
        checkpoint = ModelCheckpoint(checkpoint_path, verbose=1, save_best_only=False, save_weights_only=False, mode='auto')
        earlystopping =  EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=1, mode='auto')

        #model.fit_generator(driving_data.generate_arrays_from_file(), validation_data = (X, y), samples_per_epoch = len(y) * 4, nb_epoch=epochs, verbose = 1, callbacks=[checkpoint])
        #res=model.fit_generator(driving_data.generate_arrays_from_file(), validation_data = (X, y), samples_per_epoch = len(y) * 4, nb_epoch=epochs, verbose = 1 )
        res=model.fit_generator(driving_data.generator(driving_data.train_xs,driving_data.train_ys,256), validation_data = (X, y), samples_per_epoch = 100*256, nb_epoch=epochs, verbose = 1  ,callbacks = [ SaveModel()])

        if not os.path.exists("./outputs/sully_steering_model"):
            os.makedirs("./outputs/sully_steering_model")

        model.save_weights("./outputs/sully_steering_model/steering_angle.h5", True)
        with open('./outputs/sully_steering_model/steering_angle.json', 'w') as outfile:
          json.dump(model.to_json(), outfile)

        history=res.history
        with open('history.p','wb') as f:
           pickle.dump(history,f)

if __name__ == "__main__":
    train()
