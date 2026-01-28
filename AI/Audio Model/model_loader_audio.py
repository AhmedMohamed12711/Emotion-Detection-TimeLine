# model_loader_audio.py
"""
MULTIMODAL AUDIO EMOTION MODEL
--------------------------------
This model does 3 things:

1. Transcribes audio -> text   (Whisper)
2. Sends text -> TEXT EMOTION API
3. Runs audio emotion model (HuBERT SER)
4. Combines audio + text -> FINAL MULTIMODAL EMOTION

All outputs are Python floats/ints, zero numpy types (FastAPI-safe).
"""

import time
import requests
import numpy as np
import librosa
import whisper
from datetime import datetime
import torch
import os
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# --------------------------------------------------------
# CONFIG
# --------------------------------------------------------

TEXT_API_URL = "http://127.0.0.1:8000/emotion/text_model"   # Your text API

WHISPER_MODEL_NAME = "small"  # best speed/accuracy
whisper_model = whisper.load_model(WHISPER_MODEL_NAME)

AUDIO_MODEL_NAME = "superb/hubert-large-superb-er"
# AUDIO_MODEL_NAME = "models/facebook/wav2vec2-large-960h"
SAMPLE_RATE = 16000
SEGMENT_DURATION = 1.0
GLOBAL_WEIGHT = 0.8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_extractor = AutoFeatureExtractor.from_pretrained(AUDIO_MODEL_NAME)
audio_model = AutoModelForAudioClassification.from_pretrained(AUDIO_MODEL_NAME).to(device)

# SER Model labels (4 emotions → mapped to 7 emotions)
ser_mapping = ["anger", "joy", "neutral", "sadness"]
fusion_labels = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

emotion_category = {
    "anger": "negative",
    "disgust": "negative",
    "fear": "negative",
    "sadness": "negative",
    "joy": "positive",
    "surprise": "positive",
    "neutral": "neutral"
}


# --------------------------------------------------------
# HELPERS
# --------------------------------------------------------

def to_float(v):
    """Ensure JSON-safe Python float."""
    return float(v)


def convert_4_to_7_classes(probs4):
    """Convert HuBERT's 4 emotions to 7-class system."""
    anger = to_float(probs4[0])
    joy = to_float(probs4[1])
    neutral = to_float(probs4[2])
    sadness = to_float(probs4[3])

    disgust = anger * 0.2
    fear = sadness * 0.2
    surprise = joy * 0.2

    arr = [anger, disgust, fear, joy, neutral, sadness, surprise]
    s = sum(arr)
    return [x / s for x in arr]


def predict_audio_probs(wave, sr):
    """Predict emotion for a waveform segment."""
    inputs = feature_extractor(wave, sampling_rate=sr, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = audio_model(**inputs).logits.cpu().numpy()[0]

    exp = np.exp(logits - np.max(logits))
    probs4 = exp / exp.sum()
    return convert_4_to_7_classes(probs4)


def rms_weight(segment):
    """Energy-based weighting."""
    if segment.size == 0:
        return 1.0
    rms = np.sqrt(np.mean(segment ** 2) + 1e-12)
    w = 1.0 + float(rms * 10)
    w = max(0.3, min(w, 4.0))
    return w


# --------------------------------------------------------
# STEP 1: AUDIO -> TEXT (WHISPER)
# --------------------------------------------------------

def transcribe_audio(file_path):
    result = whisper_model.transcribe(file_path)
    return result["text"]


# --------------------------------------------------------
# STEP 2: SEND TEXT TO TEXT EMOTION API
# --------------------------------------------------------

def get_text_emotion(text):
    try:
        resp = requests.post(TEXT_API_URL, json={"text": text})
        return resp.json()
    except Exception as e:
        return None


# --------------------------------------------------------
# STEP 3: AUDIO EMOTION DETECTION
# --------------------------------------------------------

def analyze_audio_emotion(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    seg_len = int(SEGMENT_DURATION * sr)
    segments = []
    offsets = []

    i = 0
    while i < len(y):
        s = i
        e = min(i + seg_len, len(y))
        segments.append(y[s:e].astype(float))
        offsets.append(s / sr)
        if e == len(y):
            break
        i += seg_len

    combined = [0.0] * 7
    total_w = 0.0
    timeline = []

    for idx, seg in enumerate(segments):
        probs = predict_audio_probs(seg, sr)
        w = rms_weight(seg)

        for j, p in enumerate(probs):
            combined[j] += p * w

        total_w += w

        dom = int(np.argmax(probs))
        timeline.append({
            "segment_index": idx,
            "timestamp_offset": float(offsets[idx]),
            "probabilities": {fusion_labels[j]: round(float(probs[j]), 4) for j in range(7)},
            "dominant": {
                "label": fusion_labels[dom],
                "confidence": round(float(probs[dom]), 4),
                "category": emotion_category.get(fusion_labels[dom])
            },
            "intensity_weight": float(w),
            "frame_reference": f"audio_seg_{idx}"
        })

    combined = [c / total_w for c in combined]

    return {
        "timeline": timeline,
        "combined_probs": combined,
        "duration": float(duration),
        "segments_count": len(segments)
    }


# --------------------------------------------------------
# STEP 4: MULTIMODAL FUSION
# --------------------------------------------------------

def fuse_audio_text(audio_probs, text_probs):
    final = []
    for i in range(7):
        fused = (audio_probs[i] * 0.6 + text_probs[i] * 1.4) / 2.0
        final.append(float(fused))

    return final


# --------------------------------------------------------
# MAIN MULTIMODAL PIPELINE
# --------------------------------------------------------

def predict_emotion_audio(file_path):
    start = time.time()

    # 1. Transcribe
    transcribed_text = transcribe_audio(file_path)

    # 2. Text emotion
    text_result = get_text_emotion(transcribed_text)

    # 3. Audio emotion
    audio_result = analyze_audio_emotion(file_path)
    audio_probs = audio_result["combined_probs"]

    # 4. Extract text_probs (from combined_results)
    text_probs7 = [0] * 7
    if text_result and "combined_results" in text_result:
        for item in text_result["combined_results"]:
            idx = fusion_labels.index(item["label"])
            text_probs7[idx] = float(item["confidence"])
    else:
        text_probs7 = audio_probs  # fallback if text API fails

    # 5. Fuse final emotion
    fused_probs = fuse_audio_text(audio_probs, text_probs7)
    dom = int(np.argmax(fused_probs))

    processing_ms = round((time.time() - start) * 1000, 3)

    return {
        "audio_filename": os.path.basename(file_path),
        "transcribed_text": transcribed_text,

        "audio_emotion": {
            "timeline": audio_result["timeline"],
            "combined_probs": [float(x) for x in audio_result["combined_probs"]],
            "segments_count": int(audio_result["segments_count"]),
            "duration_seconds": float(audio_result["duration"])
        },

        "text_emotion": text_result,

        "final_multimodal_emotion": {
            "label": fusion_labels[dom],
            "confidence": float(fused_probs[dom]),
            "confidence_percent": round(float(fused_probs[dom]) * 100, 2),
            "category": emotion_category.get(fusion_labels[dom], "neutral")
        },

        "final_multimodal_results": [
            {
                "label": fusion_labels[i],
                "confidence": float(fused_probs[i]),
                "confidence_percent": round(float(fused_probs[i]) * 100, 2)
            } for i in range(7)
        ],

        "timestamp": datetime.now().isoformat(),
        "processing_time_ms": float(processing_ms),

        "model_info": {
            "audio_model": AUDIO_MODEL_NAME,
            "text_model_api": TEXT_API_URL,
            "whisper_model": WHISPER_MODEL_NAME,
            "fusion_version": "v1.0"
        }
    }
