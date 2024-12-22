import cv2

def preprocess_image(image_path):
    """Preprocess an image for prediction."""
    gray = cv2.cvtColor(image_path, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    normalized = resized / 255.0
    return normalized.reshape(1, 48, 48, 1)
