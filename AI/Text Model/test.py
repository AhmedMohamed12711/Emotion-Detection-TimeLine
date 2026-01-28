import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# Load model
MODEL_NAME = "cardiffnlp/twitter-roberta-base-emotion-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

labels = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]


def analyze_emotion(text):
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt")

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    # Softmax to get probabilities
    probs = F.softmax(outputs.logits, dim=1)[0]

    # Convert to Python values
    probs = probs.tolist()

    # Find dominant emotion
    dominant_idx = probs.index(max(probs))

    return {
        "text": text,
        "dominant_emotion": labels[dominant_idx],
        "confidence": max(probs),
        "all_emotions": dict(zip(labels, probs))
    }


# Example
result = analyze_emotion("I am very happy today, this is so funny!")
print(result)
