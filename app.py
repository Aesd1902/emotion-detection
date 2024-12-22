import os
import cv2
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, Response
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from sklearn.metrics import confusion_matrix

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model
MODEL_PATH = 'models/emotion_detection_model.h5'
model = load_model(MODEL_PATH)

# Emotion categories
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Route to serve index page
@app.route('/')
def index():
    return render_template('index.html')

# Route for predicting emotion from uploaded image
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(request.url)
    
    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)

    if file:
        # Save uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Preprocess the image
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (48, 48))
        image = image.astype('float32') / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        # Predict emotion
        predictions = model.predict(image)
        emotion_index = np.argmax(predictions)
        emotion_label = EMOTIONS[emotion_index]
        confidence = predictions[0][emotion_index]

        # Generate Graphs
        accuracy_graph = generate_accuracy_graph()  # Accuracy graph
        confusion_matrix_graph = generate_confusion_matrix_graph()  # Confusion matrix graph

        return render_template('result.html', 
                               emotion=emotion_label, 
                               confidence=confidence, 
                               image_path=file_path, 
                               accuracy_graph=accuracy_graph, 
                               confusion_matrix_graph=confusion_matrix_graph)

    return redirect(url_for('index'))

# Function to generate accuracy graph
def generate_accuracy_graph():
    # Example accuracy graph (replace with actual training accuracy data)
    plt.figure(figsize=(6, 4))
    epochs = list(range(1, 11))
    accuracy = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.88, 0.9, 0.92, 0.95]  # Example data
    plt.plot(epochs, accuracy, marker='o', label='Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Over Epochs')
    plt.legend()
    
    return save_plot_to_base64()

# Function to generate confusion matrix graph
def generate_confusion_matrix_graph():
    # Example confusion matrix (replace with actual predictions and labels)
    y_true = [0, 1, 2, 3, 4, 5, 6, 6, 5, 4]  # Replace with actual labels
    y_pred = [0, 1, 2, 3, 4, 5, 6, 5, 5, 4]  # Replace with predicted labels
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=EMOTIONS, yticklabels=EMOTIONS)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    return save_plot_to_base64()

# Function to save the plot as base64 for embedding in HTML
def save_plot_to_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    base64_image = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close()
    return f"data:image/png;base64,{base64_image}"

# Function to generate video stream for live emotion detection
def generate():
    cap = cv2.VideoCapture(0)  # Open webcam
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale for face detection
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

            # Draw rectangle around face and display emotion
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f"{emotion_label} ({predictions[0][emotion_index]*100:.2f}%)",
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Encode the frame to JPEG and send it for MJPEG streaming
        _, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()

        # Yield frame as part of MJPEG stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route to start live emotion detection
@app.route('/live_cam_detection')
def live_cam_detection():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
