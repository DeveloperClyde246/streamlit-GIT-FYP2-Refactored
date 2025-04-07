import streamlit as st
import os
# pip install -U openai-whisper
from services.stress_analysis_function.function_class import convert_video_to_audio, transcribe_audio, preprocess_text, remove_stopwords, convert_slang, translate_to_indonesian, stem_text, load_bert_model, predict_sentiment
import langcodes
import tempfile

import subprocess

def force_convert_webm_to_mp4(input_path):
    output_path = input_path.replace(".mp4", "_converted.mp4").replace(".webm", "_converted.mp4")
    command = [
        "ffmpeg", "-y", "-i", input_path,
        "-c:v", "libx264",
        "-c:a", "aac",
        "-strict", "experimental",
        output_path
    ]
    subprocess.run(command, check=True)
    return output_path

# RUN: python -m streamlit run "C:\Users\Admin\OneDrive\Desktop\speech-score\main-score.py"

st.title('Stress Detection')

# # Accept video file input
# uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi", "mkv"])

# Define the video directory
video_dir = "uploaded_videos"
uploaded_video = None

col1, col2 = st.columns([2, 5])

with col1:
    chosen_question = st.session_state.get("chosen_question", "No question selected.")
    # st.write(f"Question: {chosen_question}")

    if os.listdir(video_dir):
        for video_filename in os.listdir(video_dir):
            video_path = os.path.join(video_dir, video_filename)
            uploaded_video = video_path
        st.video(uploaded_video)

with col2:
    if uploaded_video is not None:

        # Single spinner for the entire process
        with st.spinner("Processing your request, please wait..."):
            # Save the uploaded file to a temporary location
            with open(uploaded_video, 'rb') as file:
                file_content = file.read()

            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(file_content)
            tfile.close()

            # Convert video to audio
            try:
                converted_path = force_convert_webm_to_mp4(tfile.name)
                os.remove(tfile.name)
            except Exception as e:
                st.error(f"❌ FFmpeg conversion failed: {e}")
                st.stop()

            # Now extract audio from the converted file
            audio_file = convert_video_to_audio(converted_path)
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

                # Preprocess text
                speech_text = preprocess_text(speech_text)
                speech_text = remove_stopwords(speech_text)
                speech_text = convert_slang(speech_text)
                speech_text = stem_text(speech_text)

                # Clean up audio file
                os.remove(audio_file)

                # Load BERT model and predict sentiment
                bert_model_name = 'bert-base-uncased'
                num_classes = 2
                model_path = "C:/Users/KEYU/Documents/GitHub/GIT-FYP2-Refactored/Prototype/models/stress_analysis_model/bert_classifier.pth"
                loaded_model, tokenizer, device = load_bert_model(bert_model_name, num_classes, model_path)
                predicted_stress = predict_sentiment(speech_text, loaded_model, tokenizer, device)

                # Show Output
                if predicted_stress == 1:
                    st.markdown(
                        """
                        <div>
                            <h5> The candidate is experiencing stress </h5>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        """
                        <div>
                            <h5> The candidate is not experiencing stress </h5>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                # 🧠 Final Score
                final_score = 70 if predicted_stress == 1 else 100
                st.markdown(f"##### 🧠 Final Candidate Score in Stress Detection: **{final_score} / 100**")

            else:
                st.error("Failed to extract audio from the video.")
    else:
        st.warning("Please upload a video file.")

if st.button("Back"):
    st.switch_page("pages/1_Home.py")


