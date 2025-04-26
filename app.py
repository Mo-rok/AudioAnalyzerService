import os
import librosa
import numpy as np
from transformers import pipeline
from flask import Flask, request, jsonify
from pathlib import Path
import uuid
from huggingface_hub import login
from dotenv import load_dotenv

# Загружаем переменные окружения из .env файла
load_dotenv()

# Получаем параметры из переменных окружения
HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')
FLASK_HOST = os.getenv('FLASK_HOST', '0.0.0.0')
FLASK_PORT = int(os.getenv('FLASK_PORT', '5000'))
FLASK_DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() in ('true', '1', 't')

# Авторизация в Hugging Face
login(HUGGINGFACE_TOKEN)

app = Flask(__name__)
AUDIO_DIR = Path("audio")
AUDIO_DIR.mkdir(exist_ok=True)

EMOTION_WEIGHTS = {
    'angry': 0.9,
    'disgust': 0.7,
    'fear': 0.8,
    'happy': 0.1,
    'sad': 0.6,
    'surprise': 0.5,
    'neutral': 0.3
}

classifier = pipeline(
    "audio-classification",
    model="xbgoose/hubert-large-speech-emotion-recognition-russian-dusha-finetuned",
    return_all_scores=True
)

def calculate_stress_score(emotions: dict) -> float:
    total = sum(emotions.values())
    if total == 0:
        return 0.0

    stress_score = 0.0
    for emotion, value in emotions.items():
        weight = EMOTION_WEIGHTS.get(emotion.lower(), 0.5)
        stress_score += (value / total) * weight

    return min(max(stress_score, 0.0), 1.0)

def download_audio(url: str) -> Path:
    import requests

    filename = Path(url.split("/")[-1].split("?")[0])
    if not filename.suffix.lower() in ['.wav', '.mp3', '.ogg', '.flac']:
        filename = Path(f"{uuid.uuid4().hex}.wav")

    target = AUDIO_DIR / filename
    print(f"↓ Скачиваем {url} → {target}")

    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        with open(target, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    print("✓ Файл загружен")
    return target

def analyze_audio(file_path: Path) -> dict:
    try:
        audio_data, sample_rate = librosa.load(
            file_path,
            sr=16000,
            mono=True,
            res_type='kaiser_fast'
        )

        audio_data = librosa.util.normalize(audio_data)

        results = classifier({
            "raw": audio_data,
            "sampling_rate": sample_rate
        })

        emotions = {item['label']: item['score'] for item in results}
        stress_level = calculate_stress_score(emotions)

        return {
            "stress_level": round(stress_level, 4),
            "emotions": emotions
        }

    except Exception as e:
        raise RuntimeError(f"Audio analysis error: {str(e)}")

@app.route("/analyze", methods=["GET", "POST"])
def analyze_endpoint():
    if request.is_json:
        data = request.get_json()
        url = data.get("url")
    else:
        url = request.args.get("url") or request.form.get("url")

    if not url:
        return jsonify(error="parameter 'url' not provided"), 400

    try:
        audio_path = download_audio(url)
        analysis_result = analyze_audio(audio_path)

        return jsonify({
            "file": str(audio_path.name),
            "analysis_result": analysis_result
        })

    except Exception as e:
        return jsonify(error=str(e)), 500

if __name__ == "__main__":
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)