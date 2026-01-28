# main_audio.py

import uvicorn
import os
import time
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from model_loader_audio import predict_emotion_audio

app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/emotion/audio_model")
async def audio_emotion_api(file: UploadFile = File(...)):
    if file is None:
        raise HTTPException(status_code=400, detail="No file uploaded")

    tmp = "tmp_audio"
    os.makedirs(tmp, exist_ok=True)

    path = os.path.join(tmp, f"{int(time.time()*1000)}_{file.filename}")
    with open(path, "wb") as f:
        f.write(await file.read())

    try:
        result = predict_emotion_audio(path)
        return result
    finally:
        if os.path.exists(path):
            os.remove(path)

@app.get("/")
def home():
    return {"status": "Audio Emotion API is running"}

if __name__ == "__main__":
    uvicorn.run("main_audio:app", host="0.0.0.0", port=8001)
