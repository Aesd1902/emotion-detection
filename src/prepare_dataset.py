import os
import pandas as pd
import numpy as np
from PIL import Image

# Paths
CSV_PATH = r'C:\Users\Uday Alugolu\OneDrive\Desktop\emotion-detection\dataset\fer2013.csv'
OUTPUT_DIR = r'C:\Users\Uday Alugolu\OneDrive\Desktop\emotion-detection\dataset\images'

# Mapping labels to emotion names (adjust as per fer2013 documentation)
EMOTIONS = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral"
}

# Create directories
for emotion in EMOTIONS.values():
    os.makedirs(os.path.join(OUTPUT_DIR, "train", emotion), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "validation", emotion), exist_ok=True)

# Load CSV
data = pd.read_csv(CSV_PATH)

# Process each row
for index, row in data.iterrows():
    pixels = np.array(row['pixels'].split(), dtype=np.uint8).reshape(48, 48)
    emotion = EMOTIONS[row['emotion']]
    usage = row['Usage'].lower()  # 'Training' or 'PublicTest'/'PrivateTest'

    if "train" in usage:
        subset = "train"
    else:
        subset = "validation"

    # Save image
    img = Image.fromarray(pixels)
    img_path = os.path.join(OUTPUT_DIR, subset, emotion, f"{index}.png")
    img.save(img_path)

print("Dataset prepared successfully.")
