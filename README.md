# AI Interview analysis system
 
## To run:
## 1.1 Install applications
1. Install ffmpeg, can refer to this video "https://www.youtube.com/watch?v=JR36oH35Fgg"
2. Install python version 3.10.0

## 1.2 Download file and install dependencies
1. Download project zip and extract into a folder within "C: Drive"
2. Open folder in an editor. The developer mainly use VSCode
3. Open a new terminal
4. Install dependencies by using command "pip install -r requirements.txt"

## 1.3 Change path
Models can be found in "models" directory
1. in "Prototype\services\stress_analysis_function\function_class.py", change model path of line 57, line 65 (Note: "/" is used instead of "\")
2. in "Prototype\services\tone_analysis_function\preprocess_function.py", change model path in line 238 until line 241 (Note: "\" is used)
3. in "Prototype\services\tone_analysis_function\preprocess_function.py", change model path in line 273 until line 279 (Note: "\" is used)
4. in "Prototype\pages\2_Facial-Expression-Analysis.py", change model path in line 15 
5. in "Prototype\pages\Emotion-Analysis.py", change model path in line 53 and line 57

## 1.4 Run program
1. In terminal, change directory to "Prototype" directory
2. Use command "streamlit run Upload-File.py"

## Directories
models: AI models, mainly called by services 

pages: Web pages

services: Functions and preprocessor called by pages

uploaded_videos: Temporary files, mainly for storing uploaded videos
