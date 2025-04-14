import streamlit as st
import tempfile
import numpy as np
import librosa
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
import os

# from preprocess_function import extract_audio, preprocess_audio, predict_personality
from services.tone_analysis_function.preprocess_function import *

# Streamlit app
st.title("Personality Analysis")

col1, col2 = st.columns([2, 5])  #left column is wider than the right column

with col1:
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
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")

    video_dir = "uploaded_videos"
    uploaded_file = None

    chosen_question = st.session_state.get("chosen_question", "No question selected.")
    # st.write(f"Question: {chosen_question}")
    #video
    if os.listdir(video_dir):
        for video_filename in os.listdir(video_dir):
            video_path = os.path.join(video_dir, video_filename)
            uploaded_file = video_path
        st.video(uploaded_file)
            #st.success(f"Video {video_filename} loaded successfully!")


with col2:
    if uploaded_file is not None:
        with open(uploaded_file, 'rb') as file: #save the uploaded file to a temporary location
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(file.read())  #write the contents to the temp file

            with st.spinner('Processing...'):
                #extract audio
                audiofile = extract_audio(tfile.name)
                features = preprocess_audio(audiofile)

                # Load the label encoder and feature scaler
                personality_le = joblib.load(r"C:\Users\KEYU\Documents\GitHub\GIT-FYP2-Refactored\Prototype\models\personality_model\personality_label_encoder.joblib") #############change path################
                personality_scaler = joblib.load(r"C:\Users\KEYU\Documents\GitHub\GIT-FYP2-Refactored\Prototype\models\personality_model\personality_feature_scaler.joblib") #############change path################ 

                # Predict personality traits
                personality_results = predict_personality(features, personality_scaler, personality_le)


        tab1, tab2 = st.tabs(["Extracted Audio Features", "Personality Traits Analysis"])

        with tab2:
            #####

            # Display Personality Traits results
            for model, scores in personality_results.items():
                st.write("Prediction traits: ")
                if len(personality_le.classes_) == len(scores):
                    
                    col1, col2 = st.columns([3, 5])     
                    with col1:
                        st.write(" ")
                        st.write(" ")
                        st.write(" ")
                        st.write(" ")
                        st.write(" ")
                        for trait, score in zip(personality_le.classes_, scores):
                            st.write(f"{trait}: {score * 100:.2f}% section")
                    with col2:
                        # Plot radar charts with color fill
                        fig = px.line_polar(
                            r=scores,
                            theta=personality_le.classes_,
                            line_close=True
                        )
                        fig.update_traces(fill='toself')
                        st.plotly_chart(fig)

                    # Determine the most likely personality trait
                    most_likely_trait = personality_le.classes_[np.argmax(scores)]
                    st.write(f"Results : The model predicts that this candidate is more likely to be {most_likely_trait}.")

                    # Provide detailed description based on the most likely personality trait
                    trait_descriptions = {
                        "openness": "This candidate are likely to prefer new, exciting situations. Candidate value knowledge, and friends and family are likely to describe them as curious and intellectual.",
                        "conscientiousness": "This candidate have a lot of self-discipline and exceed others’ expectations. Candidate have a strong focus and prefer order and planned activities over spontaneity.",
                        "extroversion": "Extroverts thrive in social situations. This candidate are action-oriented and appreciate the opportunity to work with others.",
                        "agreeableness": "This candidate are considerate, kind and sympathetic to others. Like to participate in group activities since they are capable at compromise and helping others.",
                        "neuroticism": "indicates anxiety and pessimism,lower means emotional stability. This measure can mean you have a more hopeful view of your circumstances. As a low-neurotic or emotionally stable person, others likely admire your calmness and resilience during challenges."
                    }
                    st.write(trait_descriptions.get(most_likely_trait, "No description available for this trait."))

                    # Display confidence and consistency metrics
                    # confidence = np.max(scores) * 100
                    # consistency = np.std(scores) * 100
                    # st.write(f"Confidence: {confidence:.2f}%")
                    # st.write(f"Consistency: {consistency:.2f}%")

                    # 🎯 Final Score Calculation
                    trait_weights = {
                        "openness": 1.0,
                        "conscientiousness": 2.0,
                        "extroversion": 1.5,
                        "agreeableness": 1.0,
                        "neuroticism": -2.0
                    }

                    raw_score = sum(score * trait_weights.get(trait, 0) for trait, score in zip(personality_le.classes_, scores))
                    normalized_score = ((raw_score + 2) / 4) * 100
                    normalized_score = round(normalized_score, 2)

                    st.markdown(f"##### 🧠 Final Candidate Personality Score: **{normalized_score} / 100**")
                else:
                    st.error("Mismatch between traits and scores. Check model output!")                    

                    ######

        with tab1:
            # Display the extracted audio features
            st.write("Extracted Audio Features: ")
            st.write(features)
            st.write("Shape of the features:", features.shape)

            col1, col2 = st.columns([1, 1]) 
            with col1:
                # Visualize the audio waveform
                st.write("Audio Waveform:")
                y, sr = librosa.load(audiofile, sr=None)
                fig, ax = plt.subplots()
                librosa.display.waveshow(y, sr=sr, ax=ax)
                ax.set(title='Waveform of the Audio')
                st.pyplot(fig)

            with col2:
                # Visualize the spectrogram
                st.write("Spectrogram:")
                fig, ax = plt.subplots()
                S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
                S_dB = librosa.power_to_db(S, ref=np.max)
                img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax)
                fig.colorbar(img, ax=ax, format='%+2.0f dB')
                ax.set(title='Mel-frequency spectrogram')
                st.pyplot(fig)

if st.button("Back"):
    st.switch_page("pages/1_Home.py")