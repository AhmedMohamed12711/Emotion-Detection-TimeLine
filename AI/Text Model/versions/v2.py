import time
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import emoji
import numpy as np

# ================================
# MODEL
# ================================
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

id2label = model.config.id2label
label_names = [id2label[i] for i in range(len(id2label))]

emotion_category = {
    "anger": "negative",
    "fear": "negative",
    "disgust": "negative",
    "sadness": "negative",
    "joy": "positive",
    "surprise": "positive",
    "neutral": "neutral"
}

# ================================
# 2. LEXICON (TUNED FOR v3.3)
# ================================
lexicon = {
    "fear": [
        "scared","terrified","anxious","panic","panicked","worry","worried","fear",
        "horror","afraid","dread","nervous","tension","stress","shaking","heart racing"
    ],
    "joy": [
        "happy","excited","joy","joyful","delighted","amazed","relieved","confident",
        "glad","pleased","smiling","positive","proud"
    ],
    "anger": [
        "angry","furious","mad","irritated","annoyed","rage","hate","stupid",
        "upset","frustrated","unprofessional","blaming","slamming"
    ],
    "sadness": [
        "sad","disappointed","hurt","unhappy","depressed","down","cry","grief",
        "disheartened","discouraged","drained","exhausted"
    ],
    "disgust": [
        "disgusting","gross","repugnant","nasty","revolting","sick","unfair","vomit",
        "sickening","negligence"
    ],
    "surprise": [
        "surprised","shocked","astonished","amazed","unexpected","froze","didn’t expect"
    ],
    "neutral": []
}

def lexicon_score(text):
    text_lower = text.lower()
    scores = {e: 0 for e in label_names}

    for emo, words in lexicon.items():
        for w in words:
            if w in text_lower:
                scores[emo] += 1

    total = sum(scores.values()) or 1
    for k in scores:
        scores[k] /= total

    return scores


# ================================
# 3. CLEANING
# ================================
def clean_text(text):
    text = text.strip().lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = emoji.demojize(text, delimiters=(" ", " "))
    return text


# ================================
# 4. PREDICTION v3.3
# ================================
def predict_emotion(text: str):
    start_time = time.time()
    cleaned = clean_text(text)

    max_length = 256
    inputs = tokenizer(
        cleaned,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    ).to(device)

    token_count = inputs["input_ids"].shape[1]
    input_was_truncated = token_count == max_length

    with torch.no_grad():
        logits = model(**inputs).logits

    # --------------------------
    # 1) Temperature (slightly stronger)
    # --------------------------
    temperature = 3.6
    logits = logits / temperature

    raw_probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    # --------------------------
    # 2) Smoothing stronger
    # --------------------------
    alpha = 0.27
    smoothed = raw_probs ** alpha
    smoothed = smoothed / smoothed.sum()

    # --------------------------
    # 3) Lexicon weight tuning v3.3
    # --------------------------
    weight_map = {
        "fear": 1.20,       # slightly higher
        "anger": 1.20,      # kept strong
        "joy": 0.50,        # reduced further
        "surprise": 0.60,   # reduced further
        "sadness": 1.00,
        "disgust": 0.85,     # slightly lowered
        "neutral": 0.85,     # slightly lowered
    }

    lex = lexicon_score(text)
    lex_weighted = {k: lex[k] * weight_map[k] for k in lex}

    lex_arr = np.array([lex_weighted[lbl] for lbl in label_names])

    # lexicon contribution slightly lowered for stability
    final_probs = 0.78 * smoothed + 0.22 * lex_arr

    final_probs = final_probs / final_probs.sum()

    # --------------------------
    # 4) Build results
    # --------------------------
    results = []
    for i, p in enumerate(final_probs):
        lbl = label_names[i]
        results.append({
            "label": lbl,
            "confidence": float(p),
            "confidence_percent": round(p * 100, 2)
        })

    results = sorted(results, key=lambda x: x["confidence"], reverse=True)
    top = results[0]

    processing_time = round((time.time() - start_time) * 1000, 3)

    return {
        "text": text,
        "input_length": len(text),
        # 3. New: Add token count and truncation flag
        "token_count": token_count,
        "input_was_truncated": input_was_truncated,
        
        "timestamp": datetime.now().isoformat(),
        "processing_time_ms": processing_time,

        "model_info": {
            "name": MODEL_NAME,
            "version": "1.1",
            "onnx": False,
            # 4. New: Add device info
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
