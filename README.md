# Dystrace: Dyslexia Data Collection Platform

<img width="1144" height="760" alt="image" src="https://github.com/user-attachments/assets/455e2e3c-66b7-4144-9605-c7b2070173b8" />
Dystrace is a web-based platform designed to collect and analyze multimodal data — including reading accuracy, speech patterns, comprehension scores, and real-time gaze tracking — to support dyslexia research and clinical diagnosis.

Dyslexia varies widely in type and severity, making it difficult to diagnose with simple tests. Dystrace enables experts to make more reliable assessments by collecting diverse signals such as users’ reading accuracy, vocal output, comprehension, and eye movement data.

When a user logs in, they enter demographic information (e.g., age, gender, education level) and read a text passage appropriate for their age group. Their reading is recorded through the microphone, and reading accuracy is evaluated by comparing it to the reference text using Levenshtein similarity. The platform also tracks eye movements and evaluates comprehension through automatically generated questions.

While Dystrace does not directly diagnose dyslexia, it builds a comprehensive dataset from multiple behavioral indicators, enabling researchers and clinicians to make better-informed analyses.


# 🚀 Key Features
- **Reading Test**  
  - Presents reading passages matched to user's age
  - Records speech and gaze data in real time
  - Measures reading accuracy (Levenshtein), speed (WPM), fixation/regression count, and cognitive load

- **Comprehension Test**  
  - Automatically generates multiple-choice questions using GPT-4o-Mini
  - Calculates and records comprehension scores

- **Clinic Locator**  
  - Interactive map powered by OpenStreetMap and Flutter Map 
  - Lists major dyslexia support centers and web resources

- **User Profile & History**  
  - User authentication with Firebase Auth 
  - Test results stored in Cloud Firestore 
  - Displays past records and statistical comparisons


# 📦 Tech Stack

- **Frontend**  
  - Flutter Web  
  - `flutter_map`, OpenStreetMap  
  - `firebase_auth`, `cloud_firestore`  
  - `flutter_widget_from_html_core` (rich text)

- **Backend**  
  - FastAPI + Uvicorn  
  - Whisper for STT and WAV conversion (`ffmpeg`)  
  - Levenshtein similarity calculation
  - GPT-4o-Mini for passage and question generation
  - CORS middleware for cross-origin access

- **Machine Learning (Research)**  
  - Random Forest model for dyslexia risk prediction (joblib)

 
# 🛠️ Setup Guide

## 1. Backend Server

### Virtual Environment & Dependencies
```                   
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```                        

### Add OpenAI API Key into .env file
* echo "OPENAI_API_KEY=sk-…" > .env

### Add Firebase Admin Key into .config_secret folder
* Place serviceAccountKey.json in the .config_secret folder

Firebase Admin key(JSON) is located in backend/.config_secret/serviceAccountKey.json 
```bash  
{
   "type": "...",
   "project_id": "...",
   "private_key_id": "...",
   "private_key": "...”,
   "client_email": "...",
   "client_id": "...",
   "auth_uri": "...”,
   "token_uri": "...",
   "auth_provider_x509_cert_url": "...",
   "client_x509_cert_url": "...",
   "universe_domain": "..."
 }
```

### Run Backend
```
cd backend
uvicorn main:app --reload
```

## 2. Frontend Server

### Run Frontend
```
cd frontend
flutter clean
flutter pub get
flutter run -d chrome
```
## 3. ml_pipeline Server

### Run ml_pipeline 
```
cd ml_pipeline
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
### Train the Dyslexia Model
* After installing dependencies, train and export the Random Forest model:
```
python train_and_save.py
```


# 📸 Main Screens

<img width="1512" height="947" alt="Screenshot 2025-10-19 at 5 14 12 AM" src="https://github.com/user-attachments/assets/c23860cf-1f46-48ca-935d-88d241b9bcbd" />
<img width="1512" height="947" alt="Screenshot 2025-10-19 at 5 14 24 AM" src="https://github.com/user-attachments/assets/1277120a-91db-4916-a59a-e885d576ae6f" />
<img width="1512" height="947" alt="Screenshot 2025-10-19 at 5 14 54 AM" src="https://github.com/user-attachments/assets/c6234a21-be96-4c36-a0e6-1fee89782883" />
<img width="1512" height="947" alt="Screenshot 2025-10-19 at 5 15 15 AM" src="https://github.com/user-attachments/assets/e2552c34-d7af-4d29-a45e-b5bb59f3c1c8" />
<img width="1512" height="947" alt="Screenshot 2025-10-19 at 5 15 34 AM" src="https://github.com/user-attachments/assets/41035812-177a-422e-8af1-c86d4f4ca2fe" />
<img width="1284" height="831" alt="KakaoTalk_Photo_2025-10-19-05-40-37" src="https://github.com/user-attachments/assets/209240e8-a4bd-4cc2-93e6-f6b38352044b" />
<img width="1512" height="947" alt="Screenshot 2025-10-19 at 5 19 14 AM" src="https://github.com/user-attachments/assets/127a317d-e6e8-4b44-aa97-f70d39decbbc" />
<img width="1705" height="983" alt="KakaoTalk_Image_2025-10-19-05-40-09_002" src="https://github.com/user-attachments/assets/4d31e5c8-e554-431e-82ce-53cea93eb394" />
<img width="1512" height="947" alt="Screenshot 2025-10-19 at 5 20 48 AM" src="https://github.com/user-attachments/assets/bc75f5cc-87e9-4233-9bbe-9e1c3454c66d" />
<img width="1512" height="947" alt="Screenshot 2025-10-19 at 5 20 14 AM" src="https://github.com/user-attachments/assets/50049388-3351-47bd-9b73-e7f4afb851da" />
<img width="1512" height="947" alt="Screenshot 2025-10-19 at 5 20 03 AM" src="https://github.com/user-attachments/assets/6e25fbcc-7ee8-490b-8ebd-aa74b576ab90" />

# 👥 Developers

| Name | Photo | Role | Affiliation | GitHub |
|------|------|------|-----------|--------|
| **Jaeyeon Won** | ![IMG_4743](https://github.com/user-attachments/assets/f4e5ddbd-1e0b-46dc-9bd4-1e5b5702e933) | Backend · WebGazer · ML Pipeline · Frontend | Computer Engineering @ UT Austin | [@wjy0524](https://github.com/wjy0524) |
| **Isaac Choi** | ![KakaoTalk_Photo_2025-08-04-02-25-29](https://github.com/user-attachments/assets/5441256f-2cf7-41d0-8b06-4ac70c9122c6)| Frontend · WebGazer · ML Pipeline | Computer Science @ UT Austin | [@isaacchoi031014](https://github.com/isaacchoi031014) |


