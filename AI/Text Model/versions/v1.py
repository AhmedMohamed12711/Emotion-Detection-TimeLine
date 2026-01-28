import time
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

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

def predict_emotion(text: str):
    start_time = time.time()
    
    # 1. New: Determine the device (GPU/CPU) and move the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Model inference
    # Note: max_length is crucial for determining if truncation occurs. 
    # The default for the michellejieli model is 512, which we will use here.
    max_length = 512
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    
    # New: Move inputs to the determined device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 2. New: Get token count and check for truncation
    token_count = inputs["input_ids"].shape[1]
    # Check if the text length (in tokens) equals the max_length, 
    # indicating potential truncation.
    input_was_truncated = token_count == max_length

    with torch.no_grad():
        logits = model(**inputs).logits
        # Ensure softmax also runs on the correct device if not done implicitly
        probs = torch.softmax(logits.cpu(), dim=1)[0].tolist() # Move back to CPU for .tolist()

    results = []
    for label, prob in zip(labels, probs):
        results.append({
            "label": label,
            "confidence": float(prob),
            "confidence_percent": round(prob * 100, 2)
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