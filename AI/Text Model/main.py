import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from model_loader import predict_emotion
from fastapi.middleware.cors import CORSMiddleware # Import CORS middleware

app = FastAPI()

# --- CORS Configuration ---
# This is the essential fix for the 'Access-Control-Allow-Origin' error.
# It allows requests from any origin (*) to access your API.
origins = [
    "*",  # Allows all origins for testing. In production, you would list specific domains.
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)
# --- End CORS Configuration ---


class TextRequest(BaseModel):
    text: str

@app.post("/emotion/text_model")
async def emotion_api(req: TextRequest):
    result = predict_emotion(req.text)
    return result

@app.get("/")
def home():
    return {"status": "Emotion API is running"}

if __name__ == "__main__":
    # Note: In production, run via an ASGI server (uvicorn/gunicorn) and not via __main__.
    uvicorn.run(app, host="0.0.0.0", port=8000)