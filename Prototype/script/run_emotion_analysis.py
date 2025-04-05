import sys

import os
import numpy as np
import pandas as pd
import time
import tempfile
import cv2
# Adjust path to include project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from services.facial_expression_recognition_function.Preprocessor import Preprocessor

import tensorflow as tf
from tensorflow.keras.models import load_model

# Load model
model_path = r"C:\Users\KEYU\Documents\GitHub\GIT-FYP2-Refactored\Prototype\models\facial_expression_model\model3.h5"
model = load_model(model_path)

# Video path to process
video_dir = "uploaded_videos"
video_files = os.listdir(video_dir)

if not video_files:
    print("No video found in the directory.")
else:
    for video_filename in video_files:
        video_path = os.path.join(video_dir, video_filename)

        print(f"Processing video: {video_filename}")
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            with open(video_path, 'rb') as video_file:
                temp_file.write(video_file.read())
            temp_video_path = temp_file.name

        # Start processing time
        start_time = time.time()

        # Preprocess video frames
        preprocessor = Preprocessor()
        preprocessed_data = preprocessor.preprocess(temp_video_path)
        print(f"Extracted {len(preprocessed_data)} frames from the video.")

        # Predict emotions
        processed_frames = np.array(preprocessed_data)
        predictions = model.predict(processed_frames)
        predicted_emotions = np.argmax(predictions, axis=1)

        # Calculate emotion counts
        emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        emotion_counts = pd.Series(predicted_emotions).value_counts().sort_index()
        emotion_counts.index = [emotion_labels[i] for i in emotion_counts.index]

        # Display results
        print("\nEmotion Distribution:")
        print(emotion_counts)

        max_emotion = emotion_counts.idxmax()
        print(f"\nFinal result: The facial expression of the candidate is {max_emotion} in this video.")

        # End processing time
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Total processing time: {total_time:.2f} seconds")

        # Clean up
        os.remove(temp_video_path)
