import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_fer2013(csv_path):
    """
    Load and preprocess the FER2013 dataset.
    Args:
        csv_path (str): Path to the `fer2013.csv` file.
    Returns:
        Tuple of numpy arrays: (x_train, x_val, y_train, y_val)
    """
    data = pd.read_csv(csv_path)
    pixels = data['pixels'].tolist()
    emotions = pd.get_dummies(data['emotion']).values  # One-hot encode labels
    
    # Preprocess images
    images = np.array([np.fromstring(p, sep=' ').reshape(48, 48) for p in pixels])
    images = images / 255.0  # Normalize pixel values
    
    # Split into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(
        images, emotions, test_size=0.2, random_state=42
    )
    
    # Add channel dimension for grayscale images
    x_train = np.expand_dims(x_train, axis=-1)
    x_val = np.expand_dims(x_val, axis=-1)
    
    return x_train, x_val, y_train, y_val
