# pip install matplotlib seaborn scikit-learn

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import backend as K

# Add these imports for metrics
from tensorflow.keras.metrics import Precision, Recall, AUC, BinaryAccuracy
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns

# ... [existing code remains the same until model compilation] ...

# Update the model compilation to include more metrics
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy', 
                      Precision(name='precision'),
                      Recall(name='recall'),
                      AUC(name='auc')])

# ... [existing code for data generators] ...

# Store the training history when fitting the model
history = model.fit(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

print("fit_generator")

# Save the model as before
model.save('model.keras')
model.save_weights('model.weights.h5')

print("saved model and weights")

# ----- Additional evaluation code -----

# Function to evaluate and print metrics on validation data
def evaluate_model(model, validation_generator, nb_validation_samples, batch_size):
    print("\n----- Model Evaluation -----")
    
    # Calculate predictions on validation set
    validation_generator.reset()
    y_pred_prob = model.predict(validation_generator, 
                            steps=nb_validation_samples // batch_size)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # Get true labels
    validation_generator.reset()
    y_true = validation_generator.classes
    
    # Calculate and print metrics
    print("\n--- Classification Report ---")
    print(classification_report(y_true[:len(y_pred)], y_pred, 
                              target_names=['cat', 'dog']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true[:len(y_pred)], y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['cat', 'dog'],
                yticklabels=['cat', 'dog'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    
    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_true[:len(y_pred)], y_pred_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    
    # Learning curves
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('learning_curves.png')
    
    # Precision-Recall curves
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['precision'], label='Training Precision')
    plt.plot(history.history['val_precision'], label='Validation Precision')
    plt.plot(history.history['recall'], label='Training Recall')
    plt.plot(history.history['val_recall'], label='Validation Recall')
    plt.title('Precision and Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig('precision_recall.png')
    
    # Print final metrics
    final_results = model.evaluate(validation_generator, 
                                  steps=nb_validation_samples // batch_size)
    metric_names = ['loss', 'accuracy', 'precision', 'recall', 'auc']
    
    print("\n--- Final Metrics ---")
    for name, value in zip(metric_names, final_results):
        print(f"{name}: {value:.4f}")

# Call the evaluation function
evaluate_model(model, validation_generator, nb_validation_samples, batch_size)

print("\nEvaluation complete. Visualization images saved.")