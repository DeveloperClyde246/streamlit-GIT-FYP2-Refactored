import streamlit as st
import os
import tempfile
import langcodes
from services.stress_analysis_function.function_class import (
    convert_video_to_audio, transcribe_audio, preprocess_text,
    remove_stopwords, convert_slang, translate_to_indonesian,
    stem_text, load_bert_model, predict_sentiment
)

st.title('Stress Detection')

video_dir = "uploaded_videos"
uploaded_video = None

# Display question
chosen_question = st.session_state.get("chosen_question", "No question selected.")
st.write(f"Question: {chosen_question}")

# Load first video found
if os.listdir(video_dir):
    for video_filename in os.listdir(video_dir):
        video_path = os.path.join(video_dir, video_filename)
        uploaded_video = video_path
        st.video(uploaded_video)
        break  # Process only the first video

if uploaded_video is not None:
    with st.spinner("Processing your request, please wait..."):
        # Save video temporarily
        with open(uploaded_video, 'rb') as file:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(file.read())

        # Extract audio
        audio_file = convert_video_to_audio(tfile.name)

        if audio_file:
            # Transcribe
            transcription_result = transcribe_audio(audio_file)
            speech_text = transcription_result["text"]
            detected_language = transcription_result["language"]

            st.write("Detected Language:", langcodes.get(detected_language).display_name())
            st.write("Transcribed Text:", speech_text)

            # Translate if needed
            if detected_language != "id":
                speech_text = translate_to_indonesian(speech_text)
                st.write("Translated Text:", speech_text)

            # Text preprocessing
            speech_text = preprocess_text(speech_text)
            speech_text = remove_stopwords(speech_text)
            speech_text = convert_slang(speech_text)
            speech_text = stem_text(speech_text)

            os.remove(audio_file)

            # Load BERT model and predict
            model_path = "C:/Users/KEYU/Documents/GitHub/GIT-FYP2-Refactored/Prototype/models/stress_analysis_model/bert_classifier.pth"
            loaded_model, tokenizer, device = load_bert_model('bert-base-uncased', 2, model_path)
            predicted_stress = predict_sentiment(speech_text, loaded_model, tokenizer, device)

            # Result + Final mark
            if predicted_stress == 1:
                st.write("Stress Detected: ✅ Yes")
                st.markdown("### 🧠 Final Candidate Stress Score: **70 / 100**")
            else:
                st.write("Stress Detected: ❌ No")
                st.markdown("##### 🧠 Final Candidate Stress Score: **100 / 100**")
        else:
            st.error("Failed to extract audio from the video.")
else:
    st.warning("Please upload a video file.")

if st.button("Back"):
    st.switch_page("pages/1_Home.py")
