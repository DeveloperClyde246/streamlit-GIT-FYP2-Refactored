import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from Preprocessor import Preprocessor
import tempfile
import os
import numpy as np
import cv2
import time

# Load model to predict
import tensorflow as tf
from tensorflow.keras.models import load_model
model = load_model('Model/model3.h5')


# Set the page title
st.title("Facial Expression Detection")

# Create two columns
col1, col2 = st.columns(2)

with col1:
    st.header("Upload Video")

    # Video upload functionality
    uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_video is not None:
        st.video(uploaded_video)
        st.success("Video uploaded successfully!")

        # Save the uploaded video to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            temp_file.write(uploaded_video.read())
            temp_video_path = temp_file.name

        # Start timing the processing
        start_time = time.time()

        # Process the uploaded video using Preprocessor
        preprocessor = Preprocessor()
        preprocessed_data = preprocessor.preprocess(temp_video_path)
        st.write(f"Extracted {len(preprocessed_data)} frames from the video.")

        # Predict emotions for each frame
        processed_frames = np.array(preprocessed_data)
        predictions = model.predict(processed_frames)
        predicted_emotions = np.argmax(predictions, axis=1)

        # End timing the processing
        end_time = time.time()
        total_time = end_time - start_time

        # Map predictions to emotion labels
        emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        emotion_counts = pd.Series(predicted_emotions).value_counts().sort_index()
        emotion_counts.index = [emotion_labels[i] for i in emotion_counts.index]

        # Display the total processing time
        st.write(f"Total processing time: {total_time:.2f} seconds")

        # Clean up the temporary file
        os.remove(temp_video_path)

with col2:
    st.header("Facial Expression Distribution")
    if uploaded_video is not None and len(predicted_emotions) > 0:
        # Create a pie chart based on the emotion counts
        fig, ax = plt.subplots()
        ax.pie(emotion_counts, labels=emotion_counts.index, autopct='%1.1f%%', startangle=90, colors=["#FF9999", "#66B2FF", "#99FF99", "#FFCC99", "#FFD700", "#87CEFA", "#90EE90"])
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

        # Display the pie chart
        st.pyplot(fig)
        st.write("### Emotion Distribution")
        #rename the dataframe columns
        emotion_counts = emotion_counts.reset_index()
        emotion_counts.columns = ['Emotions', 'Frames']
        st.table(emotion_counts)
        #display the massage the maximum emotion
        max_emotion = emotion_counts.loc[emotion_counts['Frames'].idxmax()]['Emotions']
        st.write(f"The facial expression of the candidate is {max_emotion} in this video")
    else:
        st.write("Upload a video to view the emotion distribution.")
