import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained model
MODEL_PATH = r'C:\Users\Uday Alugolu\OneDrive\Desktop\emotion-detection\models\emotion_detection_model.h5'

try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Emotion categories (same as used during training)
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load test image
TEST_IMAGE_PATH = r'C:\Users\Uday Alugolu\OneDrive\Desktop\emotion-detection\dataset\images\train\angry\0.png'  # Replace with your test image path

try:
    image = cv2.imread(TEST_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found at {TEST_IMAGE_PATH}. Please check the path.")
    print("Image loaded successfully.")
except Exception as e:
    print(f"Error loading image: {e}")
    exit()

# Preprocess the image
try:
    image = cv2.resize(image, (48, 48))  # Resize to match the input size of the model
    image = image.astype('float32') / 255.0  # Normalize pixel values
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    print("Image preprocessed successfully.")
except Exception as e:
    print(f"Error preprocessing image: {e}")
    exit()

# Predict the emotion
try:
    predictions = model.predict(image)
    emotion_index = np.argmax(predictions)
    detected_emotion = EMOTIONS[emotion_index]
    print(f"Detected Emotion: {detected_emotion}")
    print(f"Confidence Scores: {predictions[0]}")
except Exception as e:
    print(f"Error during prediction: {e}")
