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
    parts = re.split(r'[.!?]+', text)
    return [p.strip() for p in parts if p.strip()]


# -------------------------
# INTENSITY v2 – FIXED & OPTIMIZED
# -------------------------
def intensity_weight(sentence):
    s = sentence.lower()
    weight = 1.0  # base weight

    # ------------ FEAR ------------
    strong_fear = ["heart dropped", "panic", "terrified", "dread", "scared", "fear", "worried", "anxiety"]
    mild_fear   = ["uneasy", "nervous"]

    for w in strong_fear:
        if w in s:
            weight += 1.1   # reduced from 1.5
    for w in mild_fear:
        if w in s:
            weight += 0.5

    # ------------ SADNESS (BOOSTED) ------------
    strong_sadness = ["devastated", "broken", "grief", "heartbroken", "loss"]
    mild_sadness   = ["sad", "disappointed", "hurt"]

    for w in strong_sadness:
        if w in s:
            weight += 1.5   # strongest now
    for w in mild_sadness:
        if w in s:
            weight += 0.9

    # ------------ DISGUST (BOOSTED) ------------
    strong_disgust = ["revolting", "disgusting", "filth", "repugnant", "gross", "nasty"]
    mild_disgust   = ["unclean", "dirty", "messy"]

    for w in strong_disgust:
        if w in s:
            weight += 1.3
    for w in mild_disgust:
        if w in s:
            weight += 0.7

    # ------------ ANGER ------------
    strong_anger = ["furious", "rage", "angry", "irate"]
    mild_anger   = ["frustrated", "annoyed", "upset"]

    for w in strong_anger:
        if w in s:
            weight += 1.2
    for w in mild_anger:
        if w in s:
            weight += 0.7

    # ------------ SURPRISE ------------
    strong_surprise = ["shocked", "stunned"]
    mild_surprise   = ["unexpected", "surprised"]

    for w in strong_surprise:
        if w in s:
            weight += 0.9
    for w in mild_surprise:
        if w in s:
            weight += 0.5

    # ------------ RELIEF / JOY ------------
    relief_words = ["relieved", "calm"]
    for w in relief_words:
        if w in s:
            weight -= 0.5  # prevent joy smoothing

    # ------------ NEUTRAL REDUCTION (IMPORTANT!) ------------
    neutral_words = ["normal", "okay", "fine", "simple"]
    for w in neutral_words:
        if w in s:
            weight -= 0.7   # stronger neutral suppression

    # prevent negative or tiny weights
    if weight < 0.3:
        weight = 0.3

    return weight


# -------------------------
# Predict single sentence
# -------------------------
def predict_single(sentence, device, max_length=512):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits.cpu(), dim=1)[0].tolist()

    return probs


# -------------------------
# MAIN FUNCTION (FULL PIPELINE)
# -------------------------
def predict_emotion(text: str):
    start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    sentences = split_into_sentences(text)

    combined_probs = [0.0] * len(labels)
    total_weight = 0.0

    # Weighted prediction
    for s in sentences:
        probs = predict_single(s, device)
        w = intensity_weight(s)
        total_weight += w

        for i, p in enumerate(probs):
            combined_probs[i] += p * w

    # Normalize
    if total_weight > 0:
        combined_probs = [x / total_weight for x in combined_probs]

    # Build results list
    results = []
    for lbl, p in zip(labels, combined_probs):
        results.append({
            "label": lbl,
            "confidence": float(p),
            "confidence_percent": round(p * 100, 2)
        })

    results = sorted(results, key=lambda x: x["confidence"], reverse=True)
    top = results[0]

    # Token info
    max_length = 512
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    token_count = inputs["input_ids"].shape[1]
    input_was_truncated = token_count == max_length

    processing_time = round((time.time() - start_time) * 1000, 3)

    # Final return
    return {
        "text": text,
        "input_length": len(text),
        "token_count": token_count,
        "input_was_truncated": input_was_truncated,

        "timestamp": datetime.now().isoformat(),
        "processing_time_ms": processing_time,

        "model_info": {
            "name": MODEL_NAME,
            "version": "weighted-intensity-v2",
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
