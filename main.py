import os
import json
import joblib
import torch

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


@asynccontextmanager
async def lifespan(app: FastAPI):

    choices = {
        "category": "baseline",
        "priority": "baseline"
    }

    if os.path.exists("outputs/deployment_choice.json"):
        with open("outputs/deployment_choice.json") as file:
            choices.update(json.load(file))

    app.state.models = {}

    for label in ["category", "priority"]:

        model_type = choices[label]
        model_path = f"models/distilbert_{label}"

        if model_type == "distilbert" and not os.path.isdir(model_path):
            print("DistilBERT missing, using baseline for", label)
            model_type = "baseline"

        if model_type == "distilbert":

            tokenizer = AutoTokenizer.from_pretrained(model_path)

            model = AutoModelForSequenceClassification.from_pretrained(
                model_path
            )
            model.eval()

            app.state.models[label] = {
                "type": "distilbert",
                "model": model,
                "tokenizer": tokenizer
            }

        else:
            model = joblib.load(
                f"models/{label}_model.joblib"
            )

            app.state.models[label] = {
                "type": "baseline",
                "model": model
            }

    print("Models loaded successfully")
    yield


app = FastAPI(
    title="Customer Support Ticket Classifier",
    lifespan=lifespan
)


class Ticket(BaseModel):
    text: str = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value):
        if not value.strip():
            raise ValueError("Ticket text cannot be empty")
        return value


def predict_label(label, text):

    selected = app.state.models[label]
    model = selected["model"]

    if selected["type"] == "baseline":

        probabilities = model.predict_proba([text])[0]

        index = probabilities.argmax()

        prediction = model.classes_[index]
        confidence = float(probabilities[index])

    else:

        tokenizer = selected["tokenizer"]

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128
        )

        with torch.inference_mode():
            outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits, dim=1
        )[0]

        index = probabilities.argmax().item()

        prediction = model.config.id2label[index]
        confidence = probabilities[index].item()

    return prediction.title(), confidence


def classify_ticket(text):

    text = text.lower().strip()

    category, cat_conf = predict_label("category", text)
    priority, pri_conf = predict_label("priority", text)

    return {
        "category": category,
        "priority": priority,
        "confidence": round(min(cat_conf, pri_conf), 3)
    }


@app.get("/health")
def health():

    return {
        "status": "ok",
        "category_model": app.state.models["category"]["type"],
        "priority_model": app.state.models["priority"]["type"]
    }


@app.post("/predict")
def predict(ticket: Ticket):

    return classify_ticket(ticket.text)


@app.post("/predict-batch")
def predict_batch(tickets: list[Ticket]):

    if not tickets:
        raise HTTPException(
            status_code=422,
            detail="At least one ticket is required"
        )

    predictions = []

    for ticket in tickets:
        result = classify_ticket(ticket.text)
        predictions.append(result)

    return {"predictions": predictions}