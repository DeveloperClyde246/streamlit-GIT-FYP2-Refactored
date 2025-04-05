import streamlit as st
import tempfile
import numpy as np
import librosa
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
import os
from services.tone_analysis_function.preprocess_function import *


st.title("Emotion Analysis")
col1, col2 = st.columns([2, 5]) 

video_dir = "uploaded_videos"
uploaded_file = None

with col1:
    st.subheader("Video")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")

    chosen_question = st.session_state.get("chosen_question", "No question selected.")
    st.write(f"Question: {chosen_question}")

    if os.listdir(video_dir):
        for video_filename in os.listdir(video_dir):
            video_path = os.path.join(video_dir, video_filename)
            uploaded_file = video_path
            st.video(uploaded_file)
            #st.success(f"Video {video_filename} loaded successfully!")
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        with open(uploaded_file, 'rb') as video_file:
            tfile.write(video_file.read())

with col2:
    tab1, tab2 = st.tabs(["Extracted Audio Features", "Emotions Analysis"])
    with tab1:
        with st.spinner('Processing...'):

            audiofile = extract_audio(tfile.name)# Extract audio from video
            features = preprocess_audio(audiofile)# Preprocess audio

            # label encoder and feature scaler
            emotion_le_path = r"C:\Users\KEYU\Documents\GitHub\GIT-FYP2-Refactored\Prototype\models\emotion_model\emotion_label_encoder.joblib"
            emotion_le = joblib.load(emotion_le_path)

            emotion_scaler_path = r"C:\Users\KEYU\Documents\GitHub\GIT-FYP2-Refactored\Prototype\models\emotion_model\emotion_feature_scaler.joblib"
            emotion_scaler = joblib.load(emotion_scaler_path)
            
            #predict emotions
            emotion_results = predict_emotion(features, emotion_scaler, emotion_le)

            #display the extracted audio features
            st.write("Extracted Audio Features: ")
            st.write(features)
            st.write("Shape of the features:", features.shape)

            col1, col2 = st.columns([1, 1]) 
            with col1:
                # audio waveform
                st.write("Audio Waveform:")
                y, sr = librosa.load(audiofile, sr=None)
                fig, ax = plt.subplots()
                librosa.display.waveshow(y, sr=sr, ax=ax)
                ax.set(title='Waveform of the Audio')
                st.pyplot(fig)

            with col2:
                # spectrogram
                st.write("Spectrogram:")
                fig, ax = plt.subplots()
                S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
                S_dB = librosa.power_to_db(S, ref=np.max)
                img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax)
                fig.colorbar(img, ax=ax, format='%+2.0f dB')
                ax.set(title='Mel-frequency spectrogram')
                st.pyplot(fig)

    with tab2: 
        #display Emotions results
        for model, scores in emotion_results.items():
            st.subheader(model)
            if len(emotion_le.classes_) == len(scores):
                col1, col2 = st.columns([2, 4])
                with col1:
                    st.write(" ")
                    st.write(" ")
                    st.write(" ")
                    st.write(" ")
                    st.write(" ")
                    for emotion, score in zip(emotion_le.classes_, scores):
                        st.write(f"{emotion}: {score * 100:.2f}% section")
                with col2:
                    # Plot pie chart 
                    fig = px.pie(values=[s * 100 for s in scores], names=emotion_le.classes_, title=f"{model} Emotions",
                                    color=emotion_le.classes_, color_discrete_sequence=px.colors.qualitative.Plotly)#with consistent colors
                    st.plotly_chart(fig)

                most_likely_emotion = emotion_le.classes_[np.argmax(scores)]
                st.write(f"Results : The model predicts that this candidate is more likely to be {most_likely_emotion}.")

                # Display confidence and consistency metrics
                # confidence = np.max(scores) * 100
                # consistency = np.std(scores) * 100
                # st.write(f"Confidence: {confidence:.2f}%")
                # st.write(f"Consistency: {consistency:.2f}%")

                # Define emotion weights
                emotion_weights = {
                    'Happy': 2.0,
                    'Surprise': 1.0,
                    'Neutral': 1.0,
                    'Sad': -1.0,
                    'Fear': -1.0,
                    'Disgust': -2.0,
                    'Angry': -2.0
                }

                # Use the first model's results (assumption: one model)
                model_name, scores = next(iter(emotion_results.items()))
                emotions = emotion_le.classes_

                # Log for debugging
                st.write("💬 Emotion Probabilities & Weights:")
                raw_score = 0.0
                for emotion, score in zip(emotions, scores):
                    # Normalize emotion to title case for matching
                    weight = emotion_weights.get(emotion.title(), 0)
                    st.write(f"{emotion}: {score*100:.2f}% × {weight} = {score * weight:.4f}")
                    raw_score += score * weight

                st.write(f"🧮 Raw Score: {raw_score:.4f}")

                # Normalize raw score from [-2, +2] to [0, 100]
                min_possible = -2.0
                max_possible = 2.0
                normalized_score = ((raw_score - min_possible) / (max_possible - min_possible)) * 100
                normalized_score = round(normalized_score, 2)

                st.markdown(f" 🎓 Final Candidate Tone Score: **{normalized_score} / 100**")

            else:
                st.error("Mismatch between emotions and scores. Check model output!")

if st.button("Back"):
    st.switch_page("pages/1_Home.py")