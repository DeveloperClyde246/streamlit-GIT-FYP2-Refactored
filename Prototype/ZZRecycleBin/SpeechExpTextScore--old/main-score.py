import streamlit as st
import os
# pip install -U openai-whisper
from function_class import convert_video_to_audio, transcribe_audio, preprocess_text, remove_stopwords, convert_slang, translate_to_indonesian, stem_text, load_bert_model, predict_sentiment
import langcodes
import moviepy.editor


# RUN: python -m streamlit run main-score.py

st.title('Candidate Text-Based Scoring System')

# Accept video file input
uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi", "mkv"])

# Process the uploaded video
if uploaded_video is not None:
    st.video(uploaded_video)
    st.success("Video uploaded successfully!")

    # Single spinner for the entire process
    with st.spinner("Processing your request, please wait..."):
        # Convert video to audio
        audio_file = convert_video_to_audio(uploaded_video)

        if audio_file:
            # st.success("Audio extracted successfully!")

            # Transcribe audio
            transcription_result = transcribe_audio("output_audio.wav")
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
            model_path = "C:/Users/KEYU/Documents/GitHub/GIT-FYP2/SpeechExpTextScore --- LIM KAI ZHUN DORIAN/main-score.py"
            loaded_model, tokenizer, device = load_bert_model(bert_model_name, num_classes, model_path)
            predicted_stress = predict_sentiment(speech_text, loaded_model, tokenizer, device)

            # Show Output
            st.header("Presence of Stress")
            if predicted_stress == 1:
                st.markdown("### **The candidate is stress** 😓")
            else:
                st.markdown("### **The candidate is not stress** 😊")
        else:
            st.error("Failed to extract audio from the video.")
else:
    st.warning("Please upload a video file.")