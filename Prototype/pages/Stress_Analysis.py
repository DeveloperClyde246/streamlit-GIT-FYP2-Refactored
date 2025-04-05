import streamlit as st
import os
# pip install -U openai-whisper
from services.stress_analysis_function.function_class import convert_video_to_audio, transcribe_audio, preprocess_text, remove_stopwords, convert_slang, translate_to_indonesian, stem_text, load_bert_model, predict_sentiment
import langcodes
import tempfile

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
    st.write(f"Question: {chosen_question}")

    if os.listdir(video_dir):
        for video_filename in os.listdir(video_dir):
            video_path = os.path.join(video_dir, video_filename)
            uploaded_video = video_path
            st.video(uploaded_video)
            break  # Process only the first video
with col2:
    if uploaded_video is not None:

        # Single spinner for the entire process
        with st.spinner("Processing your request, please wait..."):
            # Save the uploaded file to a temporary location
            with open(uploaded_video, 'rb') as file:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(file.read())  # Write contents to the temp file

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
                        <div style="background-color: #FFDDC1; padding: 15px; border-radius: 10px; text-align: center;">
                            <h3 style="color: #E63946;"> The candidate is experiencing stress </h3>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        """
                        <div style="background-color: #C6F6D5; padding: 15px; border-radius: 10px; text-align: center;">
                            <h3 style="color: #2D6A4F;"> The candidate is not experiencing stress </h3>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.error("Failed to extract audio from the video.")
    else:
        st.warning("Please upload a video file.")

if st.button("Back"):
    st.switch_page("pages/1_Home.py")


