from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import Precision, Recall

import json
import os

vis_dir = 'visualizations'
if not os.path.exists(vis_dir):
    os.makedirs(vis_dir)
    print(f"Created directory: {vis_dir}")

# dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 5
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
    print("img data format channels_first")
else:
    input_shape = (img_width, img_height, 3)

# model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1)) # for 2 classes
#model.add(Activation('softmax'))
model.add(Activation('sigmoid'))

# Update the model compilation to include more metrics
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy', 
                      Precision(name='precision'),
                      Recall(name='recall')])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
#    class_mode='categorical')
    class_mode='binary')
print("train_generator")

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
#    class_mode='categorical')
    class_mode='binary')
print("validation_generator")

# Store the training history when fitting the model
history = model.fit(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)
print("fit_generator")

with open('history.json', 'w') as f:
    history_dict = {}
    # Convert numpy arrays to lists for JSON serialization
    for key in history.history:
        history_dict[key] = [float(i) for i in history.history[key]]
    json.dump(history_dict, f)

# Save the model as before
model.save('model.keras')
model.save_weights('model.weights.h5')
print("saved model and weights")