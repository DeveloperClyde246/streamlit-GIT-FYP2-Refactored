import streamlit as st
import tempfile
import numpy as np
import pandas as pd
import os
import time
import joblib
import langcodes

from services.facial_expression_recognition_function.Preprocessor import Preprocessor
from services.tone_analysis_function.preprocess_function import (
    extract_audio, preprocess_audio, predict_emotion, predict_personality
)
from services.stress_analysis_function.function_class import (
    convert_video_to_audio, transcribe_audio, preprocess_text,
    remove_stopwords, convert_slang, translate_to_indonesian,
    stem_text, load_bert_model, predict_sentiment
)
from tensorflow.keras.models import load_model

# Setup
st.title("Comprehensive Interview Analysis")
video_dir = "uploaded_videos"
uploaded_video = None

# Display question
chosen_question = st.session_state.get("chosen_question", "No question selected.")
st.write(f"Question: {chosen_question}")

# Load video
if os.listdir(video_dir):
    for video_filename in os.listdir(video_dir):
        video_path = os.path.join(video_dir, video_filename)
        uploaded_video = video_path
        st.video(uploaded_video)
        break

final_scores = {}

if uploaded_video is not None:
    with st.spinner("Processing full analysis..."):
        with open(uploaded_video, 'rb') as video_file:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(video_file.read())
            temp_path = tfile.name
            tfile.close()

        # ---------- Tone/Emotion Analysis ----------
        audiofile = extract_audio(temp_path)
        features = preprocess_audio(audiofile)

        emotion_le_path = r"C:\Users\KEYU\Documents\GitHub\GIT-FYP2-Refactored\Prototype\models\emotion_model\emotion_label_encoder.joblib"
        emotion_scaler_path = r"C:\Users\KEYU\Documents\GitHub\GIT-FYP2-Refactored\Prototype\models\emotion_model\emotion_feature_scaler.joblib"

        emotion_le = joblib.load(emotion_le_path)
        emotion_scaler = joblib.load(emotion_scaler_path)

        emotion_results = predict_emotion(features, emotion_scaler, emotion_le)
        emotion_weights = {
            'Happy': 2.0, 'Surprise': 1.0, 'Neutral': 1.0,
            'Sad': -1.0, 'Fear': -1.0, 'Disgust': -2.0, 'Angry': -2.0
        }

        for model, scores in emotion_results.items():
            raw_score = sum(score * emotion_weights.get(emotion.title(), 0)
                            for emotion, score in zip(emotion_le.classes_, scores))
            tone_score = round(((raw_score + 2) / 4) * 100, 2)
            final_scores["Tone"] = tone_score
            break  # Use first model only

        # ---------- Facial Expression Analysis ----------
        model_path = r"C:\Users\KEYU\Documents\GitHub\GIT-FYP2-Refactored\Prototype\models\facial_expression_model\model3.h5"
        facial_model = load_model(model_path)

        preprocessor = Preprocessor()
        preprocessed_data = preprocessor.preprocess(temp_path)
        predictions = facial_model.predict(np.array(preprocessed_data))
        predicted_emotions = np.argmax(predictions, axis=1)

        emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        emotion_counts = pd.Series(predicted_emotions).value_counts().sort_index()
        emotion_counts.index = [emotion_labels[i] for i in emotion_counts.index]

        facial_weights = emotion_weights  # same weights
        total_frames = emotion_counts.sum()
        facial_raw = sum(emotion_counts[emotion] * facial_weights.get(emotion, 0)
                         for emotion in emotion_counts.index) / total_frames
        facial_score = round(((facial_raw + 2) / 4) * 100, 2)
        final_scores["Facial"] = facial_score

        # ---------- Personality Analysis ----------
        personality_le = joblib.load(
            r"C:\Users\KEYU\Documents\GitHub\GIT-FYP2-Refactored\Prototype\models\personality_model\personality_label_encoder.joblib")
        personality_scaler = joblib.load(
            r"C:\Users\KEYU\Documents\GitHub\GIT-FYP2-Refactored\Prototype\models\personality_model\personality_feature_scaler.joblib")

        personality_results = predict_personality(features, personality_scaler, personality_le)
        personality_weights = {
            "openness": 1.0, "conscientiousness": 2.0, "extroversion": 1.5,
            "agreeableness": 1.0, "neuroticism": -2.0
        }

        for model, scores in personality_results.items():
            raw_score = sum(score * personality_weights.get(trait, 0)
                            for trait, score in zip(personality_le.classes_, scores))
            personality_score = round(((raw_score + 2) / 4) * 100, 2)
            final_scores["Personality"] = personality_score
            break

        # ---------- Stress Detection ----------
        audio_file = convert_video_to_audio(temp_path)
        transcription_result = transcribe_audio(audio_file)
        speech_text = transcription_result["text"]
        detected_language = transcription_result["language"]

        if detected_language != "id":
            speech_text = translate_to_indonesian(speech_text)

        speech_text = preprocess_text(speech_text)
        speech_text = remove_stopwords(speech_text)
        speech_text = convert_slang(speech_text)
        speech_text = stem_text(speech_text)

        model_path_stress = r"C:\Users\KEYU\Documents\GitHub\GIT-FYP2-Refactored\Prototype\models\stress_analysis_model\bert_classifier.pth"
        bert_model, tokenizer, device = load_bert_model("bert-base-uncased", 2, model_path_stress)
        predicted_stress = predict_sentiment(speech_text, bert_model, tokenizer, device)

        stress_score = 70 if predicted_stress == 1 else 100
        final_scores["Stress"] = stress_score

        # ---------- Final Output ----------
        st.markdown("## ✅ Component Scores:")
        for key, value in final_scores.items():
            st.write(f"{key} Score: {value} / 100")

        overall = round(sum(final_scores.values()) / len(final_scores), 2)
        st.markdown(f"### 🏆 Final Average Interview Score: **{overall} / 100**")

        # Cleanup
        os.remove(temp_path)
        if os.path.exists(audio_file):
            os.remove(audio_file)
else:
    st.warning("Please upload a video to proceed.")
