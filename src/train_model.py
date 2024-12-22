import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths
DATASET_PATH = r'C:\Users\Uday Alugolu\OneDrive\Desktop\emotion-detection\dataset\fer2013.csv'
  # Update with your dataset directory
MODEL_SAVE_PATH = r'C:\Users\Uday Alugolu\OneDrive\Desktop\emotion-detection\models\emotion_detection_model.h5'


# Data Preparation
datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    directory=r'C:\Users\Uday Alugolu\OneDrive\Desktop\emotion-detection\dataset\images\train',
    target_size=(48, 48),
    color_mode="grayscale",
    batch_size=64,
    class_mode='categorical'
)

val_data = datagen.flow_from_directory(
    directory=r'C:\Users\Uday Alugolu\OneDrive\Desktop\emotion-detection\dataset\images\validation',
    target_size=(48, 48),
    color_mode="grayscale",
    batch_size=64,
    class_mode='categorical'
)


# Build the Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  # 7 emotion classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the Model
model.fit(train_data, validation_data=val_data, epochs=25)

# Save the Model
os.makedirs('../models', exist_ok=True)
model.save(MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")
