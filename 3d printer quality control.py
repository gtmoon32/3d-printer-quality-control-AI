import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight  # For handling class imbalance
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Function to load and preprocess images
def load_and_preprocess_images(directory, label_mapping, image_width, image_height):
    # ... (Code for loading and preprocessing images)

# Directory containing your images
data_directory = "path/to/your/images"

# Label mapping for your classes
label_mapping = {
    'high_temperature': 0,
    'low_temperature': 1,
    'high_speed': 2,
    'low_speed': 3,
    'complete_fail': 4
}

# Image dimensions
your_image_width = 224
your_image_height = 224

# Load and preprocess images
images, labels = load_and_preprocess_images(data_directory, label_mapping, your_image_width, your_image_height)

# Address class imbalance (if necessary)
class_weights = class_weight.compute_class_weight('balanced', np.unique(labels), labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Data augmentation with diverse strategies
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.8, 1.2],
    contrast_range=[0.8, 1.2],
    # Consider elastic transformations or normalization augmentation
)

# Transfer learning with a deeper model
base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(your_image_width, your_image_height, 3))
model = models.Sequential()
model.add(base_model)
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))  # Regularization with dropout
model.add(layers.Dense(5, activation='softmax'))

# Freeze base model layers (optional)
for layer in base_model.layers:
    layer.trainable = False

# Compile with different optimizer, learning rate, and class weights
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(patience=5)

# Train the model
history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    epochs=20,  # Adjust epochs as needed
                    validation_data=(X_test, y_test),
                    callbacks=[early_stopping],
                    class_weight=class_weights)  # Use class weights if applicable

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')

# Save the model for later use
model.save('3d_printing_model.h5')
