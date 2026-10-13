import json
import time
import joblib
import torch
import pandas as pd

from torch.utils.data import Dataset
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

train = pd.read_csv("data/train.csv")
validation = pd.read_csv("data/validation.csv")
test = pd.read_csv("data/test.csv")

MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class TicketDataset(Dataset):

    def __init__(self, dataframe, label, labels):
        self.texts = tokenizer(
            dataframe["text"].tolist(),
            truncation=True,
            padding=True,
            max_length=128
        )

        self.targets = [
            labels.index(value)
            for value in dataframe[label]
        ]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        item = {
            key: torch.tensor(value[index])
            for key, value in self.texts.items()
        }

        item["labels"] = torch.tensor(self.targets[index])
        return item


def get_scores(actual, predicted):
    return (
        accuracy_score(actual, predicted),
        f1_score(actual, predicted, average="macro")
    )


def bert_predict(model, text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    model.eval()

    with torch.inference_mode():
        output = model(**inputs)

    index = output.logits.argmax(dim=1).item()
    return model.config.id2label[index]


def measure_speed(predict_function):

    sample = validation["text"].head(20).tolist()

    predict_function(sample[0])

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()

    for text in sample:
        predict_function(text)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    total_time = time.perf_counter() - start
    return total_time * 1000 / len(sample)


validation_rows = []
test_rows = []
choices = {}


for label in ["category", "priority"]:

    print("\nlabel.upper()--")

    labels = sorted(train[label].unique().tolist())

    train_data = TicketDataset(train, label, labels)
    val_data = TicketDataset(validation, label, labels)
    test_data = TicketDataset(test, label, labels)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(labels),
        id2label={i: name for i, name in enumerate(labels)},
        label2id={name: i for i, name in enumerate(labels)}
    )

    args = TrainingArguments(
        output_dir=f"models/checkpoints_{label}",
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        eval_strategy="epoch",
        save_strategy="no",
        report_to="none",
        logging_strategy="no",
        seed=42
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=val_data,
        processing_class=tokenizer
    )

    start = time.perf_counter()
    trainer.train()
    bert_train_time = time.perf_counter() - start

    model_path = f"models/distilbert_{label}"
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)

    print("DistilBERT model saved:", model_path)

    val_result = trainer.predict(val_data)
    test_result = trainer.predict(test_data)

    val_bert = [
        labels[i]
        for i in val_result.predictions.argmax(axis=1)
    ]

    test_bert = [
        labels[i]
        for i in test_result.predictions.argmax(axis=1)
    ]

    baseline = joblib.load(
        f"models/{label}_model.joblib"
    )

    baseline_copy = clone(baseline)

    start = time.perf_counter()
    baseline_copy.fit(train["text"], train[label])
    baseline_train_time = time.perf_counter() - start

    val_baseline = baseline.predict(validation["text"])
    test_baseline = baseline.predict(test["text"])

    baseline_speed = measure_speed(
        lambda text: baseline.predict([text])[0]
    )

    bert_speed = measure_speed(
        lambda text: bert_predict(model, text)
    )

    for split_name, dataframe, base_pred, bert_pred in [
        ("validation", validation, val_baseline, val_bert),
        ("test", test, test_baseline, test_bert)
    ]:

        base_acc, base_f1 = get_scores(
            dataframe[label], base_pred
        )

        bert_acc, bert_f1 = get_scores(
            dataframe[label], bert_pred
        )

        rows = validation_rows if split_name == "validation" else test_rows

        rows.append({
            "Model": f"{label.title()} TF-IDF",
            "Accuracy": round(base_acc, 3),
            "Macro F1": round(base_f1, 3),
            "Training time (s)": round(baseline_train_time, 3),
            "Inference time per ticket (ms)": round(baseline_speed, 3)
        })

        rows.append({
            "Model": f"{label.title()} DistilBERT",
            "Accuracy": round(bert_acc, 3),
            "Macro F1": round(bert_f1, 3),
            "Training time (s)": round(bert_train_time, 3),
            "Inference time per ticket (ms)": round(bert_speed, 3)
        })

    actual = validation[label].to_numpy()

    better = (
        (val_baseline != actual) &
        (pd.Series(val_bert).to_numpy() == actual)
    ).sum()

    worse = (
        (val_baseline == actual) &
        (pd.Series(val_bert).to_numpy() != actual)
    ).sum()

    print("\nDistilBERT corrected baseline mistakes:", better)
    print("DistilBERT introduced new mistakes:", worse)

    base_f1 = f1_score(
        actual, val_baseline, average="macro"
    )

    bert_f1 = f1_score(
        actual, val_bert, average="macro"
    )

    if bert_f1 > base_f1 + 0.03 and bert_speed <= baseline_speed * 10:
        choices[label] = "distilbert"
    else:
        choices[label] = "baseline"

    print("Selected for deployment:", choices[label])


validation_table = pd.DataFrame(validation_rows)
test_table = pd.DataFrame(test_rows)

validation_table.to_csv(
    "outputs/model_comparison.csv", index=False
)

test_table.to_csv(
    "outputs/test_comparison.csv", index=False
)

with open("outputs/deployment_choice.json", "w") as file:
    json.dump(choices, file, indent=4)

print("\nVALIDATION COMPARISON--")
print(validation_table.to_string(index=False))

print("\nTEST COMPARISON--")
print(test_table.to_string(index=False))

print("\nSelected models:", choices)