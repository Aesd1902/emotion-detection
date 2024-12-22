import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the model
MODEL_PATH = r'C:\Users\Uday Alugolu\OneDrive\Desktop\emotion-detection\models\emotion_detection_model.h5'
model = load_model(MODEL_PATH)

# Emotion categories
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Start webcam feed
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale and preprocess
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = gray_frame[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face.astype('float32') / 255.0
        face = img_to_array(face)
        face = np.expand_dims(face, axis=0)

        # Predict emotion
        predictions = model.predict(face)
        emotion_index = np.argmax(predictions)
        emotion_label = EMOTIONS[emotion_index]

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f"{emotion_label} ({predictions[0][emotion_index]*100:.2f}%)",
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow("Emotion Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import time
from flask import Response

def generate():
    # Open the webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Here you would process the frame for emotion detection
        # For example, you can add the emotion detection code here.

        # Encode the frame as JPEG
        _, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()

        # Return the frame as part of a multipart response (MJPEG stream)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

