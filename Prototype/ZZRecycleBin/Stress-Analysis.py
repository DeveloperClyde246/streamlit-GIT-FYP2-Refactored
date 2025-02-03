import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
# from Preprocessor import Preprocessor
import tempfile
import os
import numpy as np
import cv2
import time
from facial_expression_recognition.Preprocessor import Preprocessor

# Load model to predict
import tensorflow as tf
from tensorflow.keras.models import load_model
model = load_model('Model/model3.h5')

st.set_page_config(layout="wide")

st.title("Stress Detection")

col1, col2 = st.columns([2, 5])

with col1:
    st.subheader("Video")

    # Video
    video_dir = "uploaded_videos"
    uploaded_video = None

    st.write("Question: Apa yang anda ketahui tentang Ionic?")
    if os.listdir(video_dir):
        for video_filename in os.listdir(video_dir):
            video_path = os.path.join(video_dir, video_filename)
            uploaded_video = video_path
            st.video(uploaded_video)
            #st.success(f"Video {video_filename} loaded successfully!")

    if uploaded_video is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            with open(uploaded_video, 'rb') as video_file:
                temp_file.write(video_file.read())
            temp_video_path = temp_file.name

    with st.spinner('Processing...'):

        st.write("Transcript: ")
        st.write("Ionic adalah framework yang membangun aplikasi mobile dengan menggunakan html css dan javascript")

with col2:
    st.subheader("Stress Detection")
    st.write("Detected Language: Indonesia ")
    st.write("The candidate is not showing stress ")

if st.button("Back"):
    st.switch_page("pages/1_Home.py")