import streamlit as st 
import re
import os
import whisper
from googletrans import Translator
import json 
import torch
import moviepy.editor as mp
from transformers import BertTokenizer
from models.stress_analysis_model.model_class import BERTClassifier
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from moviepy.editor import VideoFileClip

# Function to convert video to audio
def convert_video_to_audio(video_file_path, output_audio_name="output_audio.wav"):
    try:
        # Use moviepy to extract audio
        video = VideoFileClip(video_file_path)
        audio = video.audio
        audio.write_audiofile(output_audio_name)

        # Close the video file and release resources
        video.close()

        return output_audio_name

    except Exception as e:
        st.error(f"Error occurred during audio extraction: {str(e)}")
        return None

# Whisper transcription
def transcribe_audio(audio_path):
    model = whisper.load_model("base")  # Change model size if needed
    result = model.transcribe(audio_path)
    return result

# Preprocess text
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove digits
    text = re.sub(r'\d+', '', text)
    
    # Remove special characters (keeping only alphabets and spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Remove Indonesian stopwords
def remove_stopwords(text, stopwords_path="C:/Users/KEYU/Documents/GitHub/GIT-FYP2/Prototype/speech_score/stopwords-id.json"):
    with open(stopwords_path, "r") as f:
        stopwords = json.load(f)
    words = text.split()
    filtered_words = [word for word in words if word not in stopwords]
    return ' '.join(filtered_words)

# Convert Indonesian slang words
def convert_slang(text, slang_path="C:/Users/KEYU/Documents/GitHub/GIT-FYP2/Prototype/speech_score/combined_slang_words.txt"):
    # Load the JSON file
    with open(slang_path, "r", encoding="utf-8") as f:
        slang_dict = json.load(f)  # Load as a dictionary

    # Replace words in the text based on the slang dictionary
    words = text.split()
    converted_words = [slang_dict.get(word, word) for word in words]
    return ' '.join(converted_words)

# Translate text to Indonesian
def translate_to_indonesian(text):
    try:
        translator = Translator()
        translated = translator.translate(text, dest="id")
        
        if translated is None:
            raise ValueError("Translation failed. Response was None.")
        
        return translated.text

    except Exception as e:
        print(f"Error during translation: {e}")
        return None
    
def stem_text(text):
    # Create the stemmer
    stem_factory = StemmerFactory()
    stemmer = stem_factory.create_stemmer()

    # Tokenize the text (split into words)
    tokens = text.split()
    
    # Apply stemming to the tokens
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    
    # Return the stemmed text (joined back into a string)
    return ' '.join(stemmed_tokens)

def load_bert_model(bert_model_name, num_classes, model_path):
    # Instantiate the tokenizer
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    
    # Determine the device (GPU or CPU)
    device = torch.device('cpu')  # Default to CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')  # Use GPU if available
    
    # Instantiate the model
    model = BERTClassifier(bert_model_name, num_classes).to(device)
    
    # Load the state dictionary with weights_only=True
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    return model, tokenizer, device

def predict_sentiment(text, model, tokenizer, device, max_length=128):
    # Set model to evaluation mode
    model.eval()

    # Tokenize and encode the input text
    encoding = tokenizer(
        text, 
        return_tensors='pt', 
        max_length=max_length, 
        padding='max_length', 
        truncation=True
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)  # Get the predicted class index

    # Return the sentiment label
    return preds.item()