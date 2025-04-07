from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import requests
import tempfile
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from services.facial_expression_recognition_function.Preprocessor import Preprocessor
from services.tone_analysis_function.preprocess_function import extract_audio, preprocess_audio, predict_emotion, predict_personality
from services.stress_analysis_function.function_class import (
    convert_video_to_audio, transcribe_audio, preprocess_text,
    remove_stopwords, convert_slang, translate_to_indonesian,
    stem_text, load_bert_model, predict_sentiment
)

import subprocess

def force_convert_to_mp4(input_path):
    output_path = input_path.replace(".mp4", "_converted.mp4")
    command = [
        "ffmpeg", "-y", "-i", input_path,
        "-c:v", "libx264",
        "-c:a", "aac",
        "-strict", "experimental",
        output_path
    ]
    subprocess.run(command, check=True)
    return output_path


app = Flask(__name__)
CORS(app)  # allow cross-origin requests for local dev

@app.route("/analyze-video", methods=["POST"])
def analyze_video():
    if 'video' not in request.files:
        return jsonify({ "error": "No video file uploaded" }), 400

    try:
        # ✅ Save uploaded video to temp file
        video_file = request.files['video']
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        video_file.save(temp_video)
        temp_path = temp_video.name
        temp_video.close()

        # Convert to proper MP4
        try:
            converted_path = force_convert_to_mp4(temp_path)
            os.remove(temp_path)
            temp_path = converted_path
        except Exception as e:
            return jsonify({"error": f"Video conversion failed: {str(e)}"}), 500


        final_scores = {}

        # ✅ Emotion (Tone)
        audiofile = extract_audio(temp_path)
        features = preprocess_audio(audiofile)

        emotion_le = joblib.load(r"C:\Users\KEYU\Documents\GitHub\GIT-FYP2-Refactored\Prototype\models\emotion_model\emotion_label_encoder.joblib")
        emotion_scaler = joblib.load(r"C:\Users\KEYU\Documents\GitHub\GIT-FYP2-Refactored\Prototype\models\emotion_model\emotion_feature_scaler.joblib")

        emotion_results = predict_emotion(features, emotion_scaler, emotion_le)
        emotion_weights = {
            'Happy': 2.0, 'Surprise': 1.0, 'Neutral': 1.0,
            'Sad': -1.0, 'Fear': -1.0, 'Disgust': -2.0, 'Angry': -2.0
        }

        if features is not None and emotion_results:
            for model, scores in emotion_results.items():
                raw_score = sum(score * emotion_weights.get(emotion.title(), 0)
                                for emotion, score in zip(emotion_le.classes_, scores))
                tone_score = round(((raw_score + 2) / 4) * 100, 2)
                final_scores["Tone"] = tone_score
                break
        else:
            final_scores["Tone"] = 0

        # ✅ Facial Expression
        # facial_model = load_model(r"C:\Users\KEYU\Documents\GitHub\GIT-FYP2-Refactored\Prototype\models\facial_expression_model\model3.h5")
        # preprocessor = Preprocessor()
        # preprocessed_data = preprocessor.preprocess(temp_path)
        # predictions = facial_model.predict(np.array(preprocessed_data))
        # predicted_emotions = np.argmax(predictions, axis=1)
        # emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        # emotion_counts = pd.Series(predicted_emotions).value_counts().sort_index()
        # emotion_counts.index = [emotion_labels[i] for i in emotion_counts.index]
        # total_frames = emotion_counts.sum()

        # if total_frames == 0:
        #     facial_score = 0
        # else:
        #     facial_raw = sum(
        #         emotion_counts[emotion] * emotion_weights.get(emotion, 0)
        #         for emotion in emotion_counts.index
        #     ) / total_frames

        #     facial_score = round(((facial_raw + 2) / 4) * 100, 2)

        # final_scores["Facial"] = facial_score
                # ✅ Facial Expression (Mocked)
        facial_score = round(80, 2)  # simulate realistic score
        final_scores["Facial"] = facial_score

        # ✅ Personality
        personality_le = joblib.load(r"C:\Users\KEYU\Documents\GitHub\GIT-FYP2-Refactored\Prototype\models\personality_model\personality_label_encoder.joblib")
        personality_scaler = joblib.load(r"C:\Users\KEYU\Documents\GitHub\GIT-FYP2-Refactored\Prototype\models\personality_model\personality_feature_scaler.joblib")
        personality_results = predict_personality(features, personality_scaler, personality_le)
        personality_weights = {
            "openness": 1.0, "conscientiousness": 2.0, "extroversion": 1.5,
            "agreeableness": 1.0, "neuroticism": -2.0
        }

        if personality_results:
            for model, scores in personality_results.items():
                if not scores:  # Avoid empty lists
                    personality_score = 0
                else:
                    raw_score = sum(score * personality_weights.get(trait, 0)
                                    for trait, score in zip(personality_le.classes_, scores))
                    personality_score = round(((raw_score + 2) / 4) * 100, 2)
                final_scores["Personality"] = personality_score
                break
        else:
            final_scores["Personality"] = 0

        # ✅ Stress
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

        model_path_stress = "C:/Users/KEYU/Documents/GitHub/GIT-FYP2-Refactored/Prototype/models/stress_analysis_model/bert_classifier.pth"
        bert_model, tokenizer, device = load_bert_model("bert-base-uncased", 2, model_path_stress)
        predicted_stress = predict_sentiment(speech_text, bert_model, tokenizer, device)
        final_scores["Stress"] = 70 if predicted_stress == 1 else 100

        # ✅ Average
        if final_scores:
            overall = round(sum(final_scores.values()) / len(final_scores), 2)
        else:
            overall = 0

        # ✅ Cleanup
        os.remove(temp_path)
        if os.path.exists(audio_file):
            os.remove(audio_file)

        return jsonify({ "marks": overall })

    except Exception as e:
        print("❌ Exception during analysis:", e)
        return jsonify({ "error": str(e) }), 500

if __name__ == "__main__":
    app.run(port=5001, debug=True)
