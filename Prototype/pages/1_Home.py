import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import os
import numpy as np
import cv2
import time

import librosa
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
from tone_analysis_dashboard.preprocess_function import *
from speech_score.function_class import convert_video_to_audio, transcribe_audio, preprocess_text, remove_stopwords, convert_slang, translate_to_indonesian, stem_text, load_bert_model, predict_sentiment
import langcodes
import tempfile

# with open('style.css') as f:
#     st.markdown(f'<style>{f.read()}<style>',unsafe_allow_html=True)

st.set_page_config(layout="wide")

# Set the page title
st.title("Interview Analysis")

# # Custom CSS to inject
# st.markdown("""
# <style>
# .streamlit-container {
#     border: 2px solid #111;
#     padding: 10px;
# }
# </style>
# """, unsafe_allow_html=True)

# Create two columns with custom width ratios
col1, col2 = st.columns([2, 5]) 

with col1:
    st.write(" ")
    st.write(" ")
    st.write(" ")

    st.subheader("Video")
    video_dir = "uploaded_videos" #vid folder
    uploaded_file = None

    chosen_question = st.session_state.get("chosen_question", "No question selected.")
    st.write(f"Question: {chosen_question}")

    if os.listdir(video_dir):# Loop through video directory
        for video_filename in os.listdir(video_dir):
            video_path = os.path.join(video_dir, video_filename)
            uploaded_file = video_path
            st.video(uploaded_file)
            #st.success(f"Video {video_filename} loaded successfully!")
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')# Save uploaded file to a temporary location
        with open(uploaded_file, 'rb') as video_file:
            tfile.write(video_file.read())
        with st.spinner('Processing...'):

            audiofile = extract_audio(tfile.name)# extract audio
            features = preprocess_audio(audiofile)# preprocess audio


    if st.button("Choose Other Video"):
            st.switch_page("Upload-File.py")

with col2: 
    tab1, tab2 = st.tabs(["Analysis", "Extracted Details"])
    with tab1:
        st.write("Frames Analysis: ")
        if st.button("Facial Expression Analysis"):
            st.switch_page("pages/2_Facial-Expression-Analysis.py")
        st.write("Tone Analysis: ")
        if st.button("Emotion Analysis"):
            st.switch_page("pages/Emotion-Analysis.py")
        if st.button("Personality Analysis"):
            st.switch_page("pages/Personality-Analysis.py")
        st.write("Transcript text Analysis: ")
        if st.button("Stress Detection"):
            st.switch_page("pages/Stress-Analysis.py")
    with tab2:
        # Convert video to audio
        audio_file = convert_video_to_audio(tfile.name)
        #st.success("Video has been processed and audio extracted!")

        if audio_file:
            # st.success("Audio extracted successfully!")

            # Transcribe audio
            transcription_result = transcribe_audio(audio_file)
            speech_text = transcription_result["text"]
            detected_language = transcription_result["language"]

            st.write("**Detected Language:**", langcodes.get(detected_language).display_name())
            st.write("**Transcribed Text:**", speech_text)

            # Translate to Indonesian if necessary
            if detected_language != "id":
                speech_text = translate_to_indonesian(speech_text)
                st.write("**Translated Text:**", speech_text)

        st.subheader("Extracted Audio Features: ")
        st.write(features)
        st.write("Shape of the features:", features.shape)

        col1, col2 = st.columns([1, 1])
        with col1:
            # Visualize the audio waveform
            st.subheader("Audio Waveform: ")
            y, sr = librosa.load(audiofile, sr=None)
            fig, ax = plt.subplots(figsize=(10, 5))  # Adjust the figure size
            librosa.display.waveshow(y, sr=sr, ax=ax)
            ax.set(title='Waveform of the Audio')
            st.pyplot(fig)

        with col2:
            # Visualize the spectrogram
            st.subheader("Spectrogram: ")
            fig, ax = plt.subplots(figsize=(10, 5))  #Adjust the figure size
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            S_dB = librosa.power_to_db(S, ref=np.max)
            img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax)
            fig.colorbar(img, ax=ax, format='%+2.0f dB')
            ax.set(title='Mel-frequency spectrogram')
            st.pyplot(fig)
