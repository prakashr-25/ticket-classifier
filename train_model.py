import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

train = pd.read_csv("data/train.csv")
validation = pd.read_csv("data/validation.csv")

X_train = train["text"]
X_val = validation["text"]

def train_model(label):

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("model", LogisticRegression(max_iter=1000))
    ])

    parameters = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "model__C": [0.1, 1, 10]
    }

    search = GridSearchCV(
        pipeline,
        parameters,
        cv=3,
        scoring="f1_macro"
    )

    search.fit(X_train, train[label])

    print("\nBest Parameters", label)
    print(search.best_params_)

    return search.best_estimator_, search.best_params_


category_model, category_params = train_model("category")

priority_model, priority_params = train_model("priority")


def evaluate(model, label, filename):

    actual = validation[label]
    predicted = model.predict(X_val)

    accuracy = accuracy_score(actual, predicted)
    macro_f1 = f1_score(actual, predicted, average="macro")

    matrix = confusion_matrix(
        actual,
        predicted,
        labels=model.classes_
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=model.classes_,
        columns=model.classes_
    )

    print("\n", label.upper(), "RESULTS")
    print("Accuracy:", round(accuracy, 3))
    print("Macro F1:", round(macro_f1, 3))
    print("\nConfusion Matrix:")
    print(matrix_df)

    matrix_df.to_csv(
        "outputs/" + filename + "_confusion_matrix.csv"
    )

    return predicted, accuracy, macro_f1

category_pred, cat_acc, cat_f1 = evaluate(
    category_model,
    "category",
    "category"
)

priority_pred, pri_acc, pri_f1 = evaluate(
    priority_model,
    "priority",
    "priority"
)

joblib.dump(
    category_model,
    "models/category_model.joblib"
)

joblib.dump(
    priority_model,
    "models/priority_model.joblib"
)

results = validation.copy()

results["predicted_category"] = category_pred
results["predicted_priority"] = priority_pred

results["category_confidence"] = (
    category_model.predict_proba(X_val).max(axis=1)
)

results["priority_confidence"] = (
    priority_model.predict_proba(X_val).max(axis=1)
)

results["wrong_labels"] = (
    (results["category"] != results["predicted_category"]).astype(int)
    +
    (results["priority"] != results["predicted_priority"]).astype(int)
)

results["confidence"] = (
    results["category_confidence"]
    + results["priority_confidence"]
) / 2

confusing = results.sort_values(
    by=["wrong_labels", "confidence"],
    ascending=[False, True]
).head(10).copy()


def guess_reason(row):

    if row["category"] != row["predicted_category"]:
        return "Ticket wording may overlap multiple categories"

    if row["priority"] != row["predicted_priority"]:
        return "Priority wording may be ambiguous"

    return "Correct prediction but comparatively low confidence"


confusing["possible_reason"] = confusing.apply(
    guess_reason,
    axis=1
)

confusing.to_csv(
    "outputs/top_10_confusing_tickets.csv",
    index=False
)

print("\nTOP 10 CONFUSING TICKETS--")
print(confusing.to_string(index=False))


#BONUS

def train_char_model(label, best_params):

    features = FeatureUnion([
        (
            "word",
            TfidfVectorizer(
                ngram_range=best_params["tfidf__ngram_range"]
            )
        ),
        (
            "character",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5)
            )
        )
    ])

    model = Pipeline([
        ("features", features),
        (
            "model",
            LogisticRegression(
                C=best_params["model__C"],
                max_iter=1000
            )
        )
    ])

    model.fit(X_train, train[label])

    return model


char_category = train_char_model(
    "category", category_params
)

char_priority = train_char_model(
    "priority", priority_params
)

char_cat_pred = char_category.predict(X_val)
char_pri_pred = char_priority.predict(X_val)

char_cat_acc = accuracy_score(
    validation["category"], char_cat_pred
)

char_cat_f1 = f1_score(
    validation["category"],
    char_cat_pred,
    average="macro"
)

char_pri_acc = accuracy_score(
    validation["priority"], char_pri_pred
)

char_pri_f1 = f1_score(
    validation["priority"],
    char_pri_pred,
    average="macro"
)


comparison = pd.DataFrame({
    "Model": [
        "Category Baseline",
        "Category + Char",
        "Priority Baseline",
        "Priority + Char"
    ],
    "Accuracy": [
        cat_acc,
        char_cat_acc,
        pri_acc,
        char_pri_acc
    ],
    "Macro_F1": [
        cat_f1,
        char_cat_f1,
        pri_f1,
        char_pri_f1
    ]
})

comparison.to_csv(
    "outputs/bonus_comparison.csv",
    index=False
)

print("\nBONUS COMPARISON--")
print(comparison.round(3).to_string(index=False))

with open("outputs/validation_report.txt", "w") as file:

    file.write("CUSTOMER SUPPORT TICKET CLASSIFIER\n\n")

    file.write("Category Best Parameters:\n")
    file.write(str(category_params) + "\n\n")

    file.write("Priority Best Parameters:\n")
    file.write(str(priority_params) + "\n\n")

    file.write(
        f"Category: Accuracy {cat_acc:.3f}, "
        f"Macro F1 {cat_f1:.3f}\n"
    )

    file.write(
        f"Priority: Accuracy {pri_acc:.3f}, "
        f"Macro F1 {pri_f1:.3f}\n\n"
    )

    file.write("BONUS COMPARISON\n")
    file.write(comparison.round(3).to_string(index=False))