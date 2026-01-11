import os
import json
import tempfile
import subprocess
import wave
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# Local modules
from transcribe import transcribe_audio
from similarity_percent import dist
from utils import compute_eye_metrics

# ----------------------------- APP SETUP -----------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
TOTAL_Q_PER_USER = 6


# ----------------------------- ML MODEL (LAZY LOAD) -----------------------------

_dyslexia_model = None

def get_dyslexia_model():
    global _dyslexia_model
    if _dyslexia_model is None:
        print("🔹 Loading dyslexia model...")
        _dyslexia_model = joblib.load("ml_pipeline/best_dyslexia_rf.joblib")
    return _dyslexia_model


# ----------------------------- MODELS -----------------------------

class UserInfo(BaseModel):
    age: int
    gender: str
    native_language: str


# ----------------------------- ENDPOINTS -----------------------------

@app.post("/get-passages")
async def get_passages(info: UserInfo):
    system_prompt = (
        "You are a professional reading passage generator for dyslexia research.\n"
        "Given user demographic info, create three age-appropriate passages.\n"
        "Return ONLY pure JSON: {\"passages\": [...]}"
    )

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(info.dict())},
        ],
    )

    try:
        data = json.loads(res.choices[0].message.content)
        return {"passages": data["passages"]}
    except Exception as e:
        raise HTTPException(500, f"GPT parsing failed: {e}")


@app.post("/reading_test")
async def reading_test(
    expected: str = Form(...),
    eye_data: str = Form(...),
    audio: UploadFile = File(...)
):
    # Save uploaded audio
    suffix = Path(audio.filename).suffix or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(await audio.read())
        input_path = f.name

    # Convert to wav
    wav_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1", wav_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        raise HTTPException(500, "Audio conversion failed")

    # Transcribe
    recognized = transcribe_audio(wav_path)

    accuracy = dist(expected.strip(), recognized.strip())
    words_read = len(recognized.split())

    with wave.open(wav_path, "rb") as wf:
        duration_seconds = int(wf.getnframes() / wf.getframerate())

    try:
        gaze_points = json.loads(eye_data)
    except Exception:
        gaze_points = []

    fixation_count, avg_fix_dur, regression_count = compute_eye_metrics(gaze_points)

    return {
        "accuracy": accuracy,
        "words_read": words_read,
        "duration_seconds": duration_seconds,
        "fixation_count": fixation_count,
        "avg_fixation_duration": avg_fix_dur,
        "regression_count": regression_count,
    }


@app.post("/get-comprehension-material")
async def get_comprehension_material(info: UserInfo):
    system_prompt = (
        "Generate 3 passages and 2 multiple-choice questions per passage.\n"
        "Return ONLY pure JSON: {\"comprehensions\": [...]}"
    )

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(info.dict())},
        ],
    )

    try:
        return json.loads(res.choices[0].message.content)
    except Exception as e:
        raise HTTPException(500, f"Parsing failed: {e}")


@app.post("/predict_dyslexia")
def predict_dyslexia(data: dict):
    model = get_dyslexia_model()

    X = [[
        data["wpm"],
        data["accuracy"],
        data["comprehension_rate"],
    ]]

    prob = model.predict_proba(X)[0][1]
    return {"probability": float(prob)}


@app.post("/final-evaluate")
def final_evaluate(
    expected: str = Form(...),
    recognized: str = Form(...),
    duration_seconds: int = Form(...),
    comprehension_correct: int = Form(...),
):
    accuracy = dist(expected.strip(), recognized.strip())
    wpm = len(recognized.split()) / (duration_seconds / 60)
    comprehension_rate = comprehension_correct / 2

    model = get_dyslexia_model()
    prob = float(model.predict_proba([[wpm, accuracy, comprehension_rate]])[0][1])

    if prob < 0.4:
        level = "Low"
    elif prob < 0.7:
        level = "Moderate"
    else:
        level = "High"

    return {
        "accuracy": round(accuracy, 2),
        "duration_seconds": duration_seconds,
        "comprehension_correct": comprehension_correct,
        "dyslexia_probability": round(prob, 3),
        "risk_level": level,
    }
