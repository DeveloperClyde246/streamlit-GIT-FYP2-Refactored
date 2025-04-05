import streamlit as st
import tempfile
import numpy as np
import pandas as pd
import os
import time
from services.facial_expression_recognition_function.Preprocessor import Preprocessor
from tensorflow.keras.models import load_model

# Load model
model_path = r"C:\Users\KEYU\Documents\GitHub\GIT-FYP2-Refactored\Prototype\models\facial_expression_model\model3.h5"
model = load_model(model_path)

st.title("Facial Expression Analysis")

video_dir = "uploaded_videos"
uploaded_video = None

chosen_question = st.session_state.get("chosen_question", "No question selected.")
st.write(f"Question: {chosen_question}")

# Load video
if os.listdir(video_dir):
    for video_filename in os.listdir(video_dir):
        video_path = os.path.join(video_dir, video_filename)
        uploaded_video = video_path
        st.video(uploaded_video)

if uploaded_video is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        with open(uploaded_video, 'rb') as video_file:
            temp_file.write(video_file.read())
        temp_video_path = temp_file.name

    with st.spinner('Processing...'):
        start_time = time.time()

        preprocessor = Preprocessor()
        preprocessed_data = preprocessor.preprocess(temp_video_path)
        processed_frames = np.array(preprocessed_data)
        predictions = model.predict(processed_frames)
        predicted_emotions = np.argmax(predictions, axis=1)

        end_time = time.time()
        total_time = end_time - start_time
        os.remove(temp_video_path)

        emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        emotion_counts = pd.Series(predicted_emotions).value_counts().sort_index()
        emotion_counts.index = [emotion_labels[i] for i in emotion_counts.index]

        # Emotion scoring weights
        emotion_weights = {
            'Happy': 2.0,
            'Surprise': 1.0,
            'Neutral': 1.0,
            'Sad': -1.0,
            'Fear': -1.0,
            'Disgust': -2.0,
            'Angry': -2.0
        }

        # Compute final score
        total_frames = emotion_counts.sum()
        raw_score = sum(emotion_counts[emotion] * emotion_weights.get(emotion, 0) for emotion in emotion_counts.index) / total_frames

        # Normalize [-2, +2] to [0, 100]
        normalized_score = ((raw_score + 2) / 4) * 100
        normalized_score = round(normalized_score, 2)

        # Output
        st.write(f"Total Processing Time: {total_time:.2f} seconds")
        st.markdown(f"### 🎓 Final Candidate Facial Expression Score: **{normalized_score} / 100**")
