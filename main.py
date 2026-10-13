from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator
import joblib

app = FastAPI(title="Customer Support Ticket Classifier")

category_model = joblib.load("models/category_model.joblib")
priority_model = joblib.load("models/priority_model.joblib")


class Ticket(BaseModel):
    text: str = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def check_text(cls, value):
        if not value.strip():
            raise ValueError("Ticket text cannot be empty")
        return value

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(ticket: Ticket):

    text = ticket.text.lower().strip()

    category = category_model.predict([text])[0]
    priority = priority_model.predict([text])[0]

    category_confidence = category_model.predict_proba(
        [text]
    ).max()

    priority_confidence = priority_model.predict_proba(
        [text]
    ).max()

    return {
        "category": category,
        "priority": priority,
        "confidence": {
            "category": round(float(category_confidence), 3),
            "priority": round(float(priority_confidence), 3)
        }
    }