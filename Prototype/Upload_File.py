import streamlit as st
import os
import requests
import subprocess

# Setup
st.title("Analyze Uploaded Video")

video_dir = "uploaded_videos"
if not os.path.exists(video_dir):
    os.makedirs(video_dir)

def clear_previous_videos():
    for file in os.listdir(video_dir):
        file_path = os.path.join(video_dir, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

def convert_webm_to_mp4(input_path):
    output_path = input_path.replace(".webm", ".mp4")
    command = [
        "ffmpeg", "-y", "-i", input_path,
        "-c:v", "libx264", "-c:a", "aac", "-strict", "experimental",
        output_path
    ]
    subprocess.run(command, check=True)
    return output_path

def force_convert_webm_to_mp4(input_path):
    output_path = input_path.replace(".webm", "_converted.mp4").replace(".mp4", "_converted.mp4")
    
    command = [
        "ffmpeg", "-y", "-i", input_path,
        "-c:v", "libx264",
        "-c:a", "aac",
        "-strict", "experimental",
        output_path
    ]
    
    subprocess.run(command, check=True)
    return output_path

# Get video URL from query params
video_url = st.query_params.get("video_url")
video_url = video_url[0] if isinstance(video_url, list) else video_url

if video_url:
    clear_previous_videos()
    original_name = "auto_downloaded.webm"
    webm_path = os.path.join(video_dir, original_name)

    try:
        # ✅ Download webm
        response = requests.get(video_url)
        with open(webm_path, "wb") as f:
            f.write(response.content)

        # ✅ Convert it
        try:
            mp4_path = force_convert_webm_to_mp4(webm_path)
            st.video(mp4_path)
            st.success(f"Video converted successfully!")
        except Exception as e:
            st.error(f"FFmpeg conversion failed: {e}")

        if st.button("Proceed"):
            st.switch_page("pages/1_Home.py")

    except Exception as e:
        st.error(f"❌ Failed to process video: {e}")

else:
    st.warning("No video URL provided. Append ?video_url=... to the Streamlit URL.")
