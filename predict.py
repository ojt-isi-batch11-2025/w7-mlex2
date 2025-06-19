from keras.models import load_model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

from keras.preprocessing.image import img_to_array, load_img
import numpy as np

# -----------------------------------------------------------------------------
# Dimensions of images
# -----------------------------------------------------------------------------
img_width, img_height = 150, 150
input_shape = (img_width, img_height, 3)

# -----------------------------------------------------------------------------
# Load model
# -----------------------------------------------------------------------------
test_model = load_model('model.keras')

# -----------------------------------------------------------------------------
# Image data path
# -----------------------------------------------------------------------------
basedir = "data/test/"

# -----------------------------------------------------------------------------
# Prediction
# -----------------------------------------------------------------------------
i=0

def predict(basedir, i, model):
    
    path = basedir+str(i)+'.JPG'
    
    img = load_img(path,"rgb",target_size=(img_width,img_height))
        
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
        
    #preds = model.predict_classes(x) #old syntax
    preds = (model.predict(x) > 0.5).astype("int32") #for binary classification
    #preds = np.argmax(model.predict(x), axis=-1) #for multi-class classification
    
    #probs = model.predict_proba(x)  #old 
    probs=model.predict(x)
    probs=(probs > 0.5).astype(int) #for binary classification
        
    if preds==0: print('cat')
    if preds==1: print('dog')

    print ('')
    #print ('Probability: ', probs*100)
    print ('Probability: ', probs)
    
    return preds, probs
 
# ***********************************************    
# MAIN
for i in range (1,21): # Images to be tested

    print('Test Sample: ', i)

    (preds,probs)=predict(basedir, i, test_model) # prediction 

    print(' ')

    
