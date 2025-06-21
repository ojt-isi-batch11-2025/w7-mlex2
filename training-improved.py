from sklearn.model_selection import RandomizedSearchCV
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import Precision, Recall
import tensorflow as tf
import numpy as np
import json
import os
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterSampler

class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(tf.greater(y_pred, 0.5), tf.float32)
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)
        
    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()
        
    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return tf.math.divide_no_nan(2 * p * r, p + r)

# Create visualization directory
vis_dir = 'visualizations'
if not os.path.exists(vis_dir):
    os.makedirs(vis_dir)
    print(f"Created directory: {vis_dir}")

# Image dimensions
img_width, img_height = 150, 150

# Data directories and parameters
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 15  # Will be used in RandomizedSearchCV
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# Define model creation function with hyperparameters as arguments
def create_model(filters_1=32, filters_2=32, filters_3=64, 
                 dense_units=64, learning_rate=0.001, dropout_rate=0.5):
    model = Sequential()
    
    # First convolutional block with batch normalization
    model.add(Conv2D(filters_1, (3, 3), input_shape=input_shape, padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Second convolutional block
    model.add(Conv2D(filters_2, (3, 3), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Third convolutional layer
    model.add(Conv2D(filters_3, (3, 3), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Fully connected layers
    model.add(Flatten())
    model.add(Dropout(dropout_rate))
    model.add(Dense(dense_units))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy', Precision(name='precision'), Recall(name='recall'), F1Score(name='f1')])
    
    return model

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# Only rescaling for validation
test_datagen = ImageDataGenerator(rescale=1./255)

# Data generators
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

# Create early stopping and checkpoint callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    verbose=1,
    restore_best_weights=True
)

model_checkpoint = ModelCheckpoint(
    'best_model_checkpoint.keras',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# Wrap the Keras model into a scikit-learn compatible object
model = KerasClassifier(
    model=create_model,
    verbose=1,
    callbacks=[early_stopping, model_checkpoint]
)

# Define hyperparameter search space with correct parameter names
param_grid = {
    'model__filters_1': [16, 32, 64],
    'model__filters_2': [16, 32, 64, 128],
    'model__filters_3': [32, 64, 128, 256],
    'model__dense_units': [32, 64, 128, 256],
    'model__learning_rate': [0.0001, 0.001, 0.01, 0.1],
    'model__dropout_rate': [0.3, 0.4, 0.5, 0.6],
    'epochs': [5, 10, 15, 20, 25, 30, 40],
    'batch_size': [10, 16, 20, 32, 64]
}

# Set up the RandomizedSearchCV
n_iter = 30  # Number of parameter settings sampled
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_grid,
    n_iter=n_iter,
    cv=3,  # Use 3-fold cross-validation
    verbose=2,
    scoring='accuracy',  # Use accuracy for scoring
    refit=True,  # Ensure the best model is refitted
    n_jobs=1,  # Use 1 job since GPU training typically doesn't benefit from parallelism
    random_state=69
)

# Prepare the data for sklearn's fit method
# Since we can't directly use data generators, we'll create small representative samples
# This is just for the search process - final model will train on full dataset
# Collect multiple batches to create a more representative dataset for hyperparameter tuning
X_batches, y_batches = [], []
num_batches = 25 # Collecting 10 batches (160 images with batch_size=16)
print(f"Collecting {num_batches} batches for hyperparameter tuning...")

for i in range(num_batches):
    X_batch, y_batch = next(train_generator)
    X_batches.append(X_batch)
    y_batches.append(y_batch)
    print(f"Collected batch {i+1}/{num_batches}")

X_train = np.concatenate(X_batches)
y_train = np.concatenate(y_batches)
print(f"Total samples for hyperparameter tuning: {len(X_train)}")

# Fit RandomizedSearchCV to find the best hyperparameters
random_search.fit(X_train, y_train)

# Print results
print("Best parameters found: ", random_search.best_params_)
print("Best score found: ", random_search.best_score_)  # Now correctly reporting F1

# Save the results to a JSON file
results = {
    'best_params': random_search.best_params_,
    'best_score': float(random_search.best_score_),
    'cv_results': {
        'params': [str(p) for p in random_search.cv_results_['params']],
        'mean_test_score': random_search.cv_results_['mean_test_score'].tolist(),
        'std_test_score': random_search.cv_results_['std_test_score'].tolist(),
        'rank_test_score': random_search.cv_results_['rank_test_score'].tolist()
    }
}

with open('randomized_search_results.json', 'w') as f:
    json.dump(results, f, indent=4)

# Now train the final model with the best parameters on the full dataset
best_params = random_search.best_params_
final_model = create_model(
    filters_1=best_params['model__filters_1'],
    filters_2=best_params['model__filters_2'],
    filters_3=best_params['model__filters_3'],
    dense_units=best_params['model__dense_units'],
    learning_rate=best_params['model__learning_rate'],
    dropout_rate=best_params['model__dropout_rate']
)
# Get batch size and epochs from best params
batch_size = best_params.get('batch_size', 16)
epochs = best_params.get('epochs', 15)

# Final model training
history = final_model.fit(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    callbacks=[early_stopping, model_checkpoint]
)

# Save the final model
final_model.save('mrs.keras')
final_model.save_weights('mrs.weights.h5')

# Save the training history
history_dict = {}
for key in history.history:
    history_dict[key] = [float(i) for i in history.history[key]]
    
with open('mrs.history.json', 'w') as f:
    json.dump(history_dict, f)

# Plot training & validation accuracy and loss values
# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc='upper left')

# plt.subplot(1, 2, 2)
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc='upper left')

# plt.tight_layout()
# plt.savefig(os.path.join(vis_dir, 'train-model_training_history.png'))

print("RandomizedSearchCV completed. Best model saved.")