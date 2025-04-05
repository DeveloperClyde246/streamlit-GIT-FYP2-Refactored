import streamlit as st
import tempfile
import numpy as np
import joblib
import os
from services.tone_analysis_function.preprocess_function import extract_audio, preprocess_audio, predict_personality

# Streamlit app
st.title("Personality Analysis")

video_dir = "uploaded_videos"
uploaded_file = None

chosen_question = st.session_state.get("chosen_question", "No question selected.")
st.write(f"Question: {chosen_question}")

# Load video
if os.listdir(video_dir):
    for video_filename in os.listdir(video_dir):
        video_path = os.path.join(video_dir, video_filename)
        uploaded_file = video_path
        st.video(uploaded_file)

# Process video
if uploaded_file is not None:
    with open(uploaded_file, 'rb') as file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(file.read())

    with st.spinner('Processing...'):
        # Audio preprocessing
        audiofile = extract_audio(tfile.name)
        features = preprocess_audio(audiofile)

        # Load encoders
        personality_le = joblib.load(r"C:\Users\KEYU\Documents\GitHub\GIT-FYP2-Refactored\Prototype\models\personality_model\personality_label_encoder.joblib")
        personality_scaler = joblib.load(r"C:\Users\KEYU\Documents\GitHub\GIT-FYP2-Refactored\Prototype\models\personality_model\personality_feature_scaler.joblib")

        # Predict
        personality_results = predict_personality(features, personality_scaler, personality_le)

        
######
        for model, scores in personality_results.items():
            if len(personality_le.classes_) == len(scores):
                st.subheader("Predicted Traits:")
                for trait, score in zip(personality_le.classes_, scores):
                    st.write(f"{trait}: {score * 100:.2f}%")

                most_likely_trait = personality_le.classes_[np.argmax(scores)]
                confidence = np.max(scores) * 100
                consistency = np.std(scores) * 100

                st.markdown(f"##### 🎯 Predicted Personality: **{most_likely_trait}**")
                # st.write(f"Confidence: {confidence:.2f}%")
                # st.write(f"Consistency: {consistency:.2f}%")

                # Trait Descriptions
                trait_descriptions = {
                    "openness": "This candidate is likely to prefer new, exciting situations. Curious and intellectual.",
                    "conscientiousness": "This candidate has self-discipline, strong focus, and prefers order.",
                    "extroversion": "This candidate thrives in social situations and enjoys collaboration.",
                    "agreeableness": "This candidate is kind, empathetic, and good at teamwork.",
                    "neuroticism": "Lower scores suggest emotional stability and calmness under pressure."
                }
                st.write("📝 Description:")
                st.write(trait_descriptions.get(most_likely_trait, "No description available."))

                # 💯 Final Score Calculation
                trait_weights = {
                    "openness": 1.0,
                    "conscientiousness": 2.0,
                    "extroversion": 1.5,
                    "agreeableness": 1.0,
                    "neuroticism": -2.0
                }

                raw_score = sum(score * trait_weights.get(trait, 0) for trait, score in zip(personality_le.classes_, scores))
                normalized_score = ((raw_score + 2) / 4) * 100  # Normalize from [-2, +2] to [0, 100]
                normalized_score = round(normalized_score, 2)

                st.markdown(f"##### 🧠 Final Candidate Personality Score: **{normalized_score} / 100**")

        #######
            else:
                st.error("Mismatch between traits and scores. Check model output!")

if st.button("Back"):
    st.switch_page("pages/1_Home.py")
