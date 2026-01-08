from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(
  
    title="Phishing Email Detection API",
    description="Detects whether an email is PHISHING or LEGIT using a DL model",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    model = tf.keras.models.load_model("phishing_lstm_model.h5")
    tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
except Exception as e:
    raise RuntimeError(f" Model or tokenizer loading failed: {e}")

MAX_LEN = 1000


class EmailRequest(BaseModel):
    email_text: str
    # FRom: str
    

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float


@app.get("/")
async def home():
    return {
        "message": " Phishing Detection API Running Successfully!"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_email(request: EmailRequest):

    try:
      
        seq = tokenizer.texts_to_sequences([request.email_text])
        padded = pad_sequences(seq, maxlen=MAX_LEN)

        pred = model.predict(padded, verbose=0)[0][0]

        label = "PHISHING" if pred >= 0.5 else "LEGIT"

        return PredictionResponse(
            prediction=label,
            confidence=round(float(pred), 4)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )



