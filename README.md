
commaai_model -- 15
https://github.com/commaai/research


sully_model -- 5 
https://github.com/jacobgil/keras-steering-angle-visualizations.git
  -- needed to turn dropout back on!
http://github.com/SullyChen/Autopilot-TensorFlow/


Notes:

Augmentation techniques
   Cropping - cropping the top and bottom didn't seem to help the driving
   Rotation - 
   Brightness - seemed to fix driving into the parking lot
   Black Squares -
   Shadows

samples_per_epoch seems to be proportional to how much augmentation is done.. otherwise it overfits
