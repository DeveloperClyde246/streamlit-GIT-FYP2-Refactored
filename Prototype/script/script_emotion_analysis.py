import streamlit as st
import tempfile
import numpy as np
import joblib
import os
from services.tone_analysis_function.preprocess_function import extract_audio, preprocess_audio, predict_emotion

st.title("Emotion Analysis")

video_dir = "uploaded_videos"
uploaded_file = None

# Display question info
chosen_question = st.session_state.get("chosen_question", "No question selected.")
st.write(f"Question: {chosen_question}")

# Display video
if os.listdir(video_dir):
    for video_filename in os.listdir(video_dir):
        video_path = os.path.join(video_dir, video_filename)
        uploaded_file = video_path
        st.video(uploaded_file)

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    with open(uploaded_file, 'rb') as video_file:
        tfile.write(video_file.read())

    with st.spinner('Processing...'):
        # Extract and preprocess audio
        audiofile = extract_audio(tfile.name)
        features = preprocess_audio(audiofile)

        # Load models
        emotion_le_path = r"C:\Users\KEYU\Documents\GitHub\GIT-FYP2-Refactored\Prototype\models\emotion_model\emotion_label_encoder.joblib"
        emotion_scaler_path = r"C:\Users\KEYU\Documents\GitHub\GIT-FYP2-Refactored\Prototype\models\emotion_model\emotion_feature_scaler.joblib"
        
        emotion_le = joblib.load(emotion_le_path)
        emotion_scaler = joblib.load(emotion_scaler_path)

        # Predict
        emotion_results = predict_emotion(features, emotion_scaler, emotion_le)

        # Display results
        for model, scores in emotion_results.items():
            st.subheader(f"Model: {model}")
            if len(emotion_le.classes_) == len(scores):
                for emotion, score in zip(emotion_le.classes_, scores):
                    st.write(f"{emotion}: {score * 100:.2f}%")

                most_likely_emotion = emotion_le.classes_[np.argmax(scores)]
                confidence = np.max(scores) * 100
                consistency = np.std(scores) * 100

                st.write(f"\n➡️ Final Result: {most_likely_emotion}")
                st.write(f"Confidence: {confidence:.2f}%")
                st.write(f"Consistency: {consistency:.2f}%")
                
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

                st.markdown(f"### 🎓 Final Candidate Tone Score: **{normalized_score} / 100**")



            else:
                st.error("Mismatch between emotions and scores. Check model output!")

if st.button("Back"):
    st.switch_page("pages/1_Home.py")
