import time
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re

MODEL_NAME = "michellejieli/emotion_text_classifier"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

labels = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

emotion_category = {
    "anger": "negative",
    "disgust": "negative",
    "fear": "negative",
    "sadness": "negative",
    "joy": "positive",
    "surprise": "positive",
    "neutral": "neutral"
}

# -------------------------
# Sentence Splitter
# -------------------------
def split_into_sentences(text):
    # Split on ., !, ?
    parts = re.split(r'[.!?]+', text)
    return [p.strip() for p in parts if p.strip()]


# -------------------------
# Predict per-sentence
# -------------------------
def predict_single(sentence, device, max_length=512):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits.cpu(), dim=1)[0].tolist()

    return probs


# -------------------------
# MAIN FUNCTION
# -------------------------
def predict_emotion(text: str):
    start_time = time.time()

    # Device config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Split into sentences
    sentences = split_into_sentences(text)

    all_sentence_probs = []
    combined_probs = [0.0] * len(labels)

    # Predict for each sentence
    for s in sentences:
        probs = predict_single(s, device)
        all_sentence_probs.append(probs)
        for i, p in enumerate(probs):
            combined_probs[i] += p

    # Average sentence probabilities
    if len(sentences) > 0:
        combined_probs = [x / len(sentences) for x in combined_probs]

    # -------------------------
    # Create Results List
    # -------------------------
    results = []
    for lbl, p in zip(labels, combined_probs):
        results.append({
            "label": lbl,
            "confidence": float(p),
            "confidence_percent": round(p * 100, 2)
        })

    results = sorted(results, key=lambda x: x["confidence"], reverse=True)
    top = results[0]

    # -------------------------
    # Token count + truncation
    # -------------------------
    max_length = 512
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    token_count = inputs["input_ids"].shape[1]
    input_was_truncated = token_count == max_length

    processing_time = round((time.time() - start_time) * 1000, 3)

    # -------------------------
    # FINAL RETURN (SAME FORMAT)
    # -------------------------
    return {
        "text": text,
        "input_length": len(text),
        "token_count": token_count,
        "input_was_truncated": input_was_truncated,

        "timestamp": datetime.now().isoformat(),
        "processing_time_ms": processing_time,

        "model_info": {
            "name": MODEL_NAME,
            "version": "1.1",
            "onnx": False,
            "device_used": str(device)
        },

        "dominant_emotion": {
            "label": top["label"],
            "confidence": top["confidence"],
            "confidence_percent": top["confidence_percent"],
            "category": emotion_category.get(top["label"], "unknown")
        },

        "results": results
    }
