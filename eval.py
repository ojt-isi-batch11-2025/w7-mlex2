import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import json
import os

# @tf.keras.saving.register_keras_serializable(package="custom_metrics")
# class F1Score(tf.keras.metrics.Metric):
#     def __init__(self, name='f1_score', **kwargs):
#         super().__init__(name=name, **kwargs)
#         self.precision = tf.keras.metrics.Precision()
#         self.recall = tf.keras.metrics.Recall()
        
#     def update_state(self, y_true, y_pred, sample_weight=None):
#         y_pred = tf.cast(tf.greater(y_pred, 0.5), tf.float32)
#         self.precision.update_state(y_true, y_pred, sample_weight)
#         self.recall.update_state(y_true, y_pred, sample_weight)
        
#     def reset_state(self):
#         self.precision.reset_state()
#         self.recall.reset_state()
        
#     def result(self):
#         p = self.precision.result()
#         r = self.recall.result()
#         return tf.math.divide_no_nan(2 * p * r, p + r)
        
#     # Add get_config method for serialization
#     def get_config(self):
#         config = super().get_config()
#         return config

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

# You also need to define these variables that are used in your functions
img_width, img_height = 150, 150
validation_data_dir = 'data/validation'
nb_validation_samples = 800
batch_size = 16

vis_dir = 'visualizations'
if not os.path.exists(vis_dir):
    os.makedirs(vis_dir)
    print(f"Created directory: {vis_dir}")

# Function to evaluate and print metrics on validation data
def evaluate_model(model_path, history_path=None):
    print(f"\n----- Model Evaluation for {model_path} -----")
    
    # Load the saved model
    custom_objects = {'F1Score': F1Score}
    model = load_model(model_path, custom_objects=custom_objects)
    print("Model loaded successfully")
    
    # Load history if available
    history = None
    if history_path and os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
        print("Training history loaded successfully")
    
    # Set up the validation data generator
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False)  # Important: don't shuffle for consistent predictions
    
    # Calculate predictions on validation set
    y_pred_prob = model.predict(validation_generator, 
                        steps=nb_validation_samples // batch_size)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # Get true labels
    validation_generator.reset()
    y_true = validation_generator.classes[:len(y_pred)]
    
    # Calculate and print metrics
    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, 
                              target_names=['cat', 'dog']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['cat', 'dog'],
                yticklabels=['cat', 'dog'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(f'{vis_dir}/eval-cm.png')
    plt.close()
    
    # If history is available, plot learning curves
    if history:
        # Learning curves
        plt.figure(figsize=(12, 5))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history['accuracy'], label='Training Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'{vis_dir}/eval-learning-curves.png')
        plt.close()
        
        # Precision-Recall curves if available
        if 'precision' in history and 'recall' in history:
            plt.figure(figsize=(8, 6))
            plt.plot(history['precision'], label='Training Precision')
            plt.plot(history['val_precision'], label='Validation Precision')
            plt.plot(history['recall'], label='Training Recall')
            plt.plot(history['val_recall'], label='Validation Recall')
            plt.title('Precision and Recall')
            plt.xlabel('Epoch')
            plt.ylabel('Score')
            plt.legend()
            plt.savefig(f'{vis_dir}/eval-precision-recall.png')
            plt.close()
    
    # Print final metrics
    print("\n--- Final Metrics ---")
    final_results = model.evaluate(validation_generator, 
                                  steps=nb_validation_samples // batch_size)
    metric_names = model.metrics_names
    
    for name, value in zip(metric_names, final_results):
        print(f"{name}: {value:.4f}")
    
    return final_results

if __name__ == "__main__":
    # Path to the saved model
    model_path = input("Enter the path to the saved model (default: 'model.keras'): ") or 'model.keras'
    
    # Path to saved history (optional) - you need to save this during training
    history_path = input("Enter the path to the saved training history (default: 'history.json'): ") or 'history.json'
    
    # Run evaluation
    evaluate_model(model_path, history_path)
    
    print("\nEvaluation complete. Visualization images saved.")