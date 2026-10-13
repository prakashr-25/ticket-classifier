import pandas as pd
from sklearn.model_selection import train_test_split

issues = {
    "billing": [
        "I was charged twice for my order",
        "My refund has not arrived",
        "The invoice amount is incorrect",
        "My payment was declined",
        "I was charged after cancellation",
        "My discount was not applied"
    ],
    "technical": [
        "The application keeps crashing",
        "The website is not loading",
        "My files are not uploading",
        "The dashboard is running slowly",
        "The software shows an error",
        "The system is not syncing"
    ],
    "account": [
        "I cannot log in to my account",
        "My password reset link is not working",
        "My account has been locked",
        "I cannot update my email address",
        "Two factor authentication is failing",
        "My profile changes are not saving"
    ]
}

priorities = {
    "low": [
        "This is not urgent",
        "I can wait a few days",
        "Please handle this when possible"
    ],
    "medium": [
        "Please resolve this soon",
        "This affects my daily work",
        "I need help within the day"
    ],
    "high": [
        "This needs immediate attention",
        "Our work has completely stopped",
        "Please treat this as an emergency"
    ]
}

contexts = [
    "I noticed this problem today",
    "It started after the latest update",
    "It happens whenever I use the service",
    "This problem has happened several times",
    "I need assistance from the support team"
]

endings = [
    "Please check and help me",
    "Please provide a solution"
]

tickets = []

for category, problems in issues.items():
    for priority, messages in priorities.items():
        for i in range(60):

            problem = problems[i % 6]
            context = contexts[(i // 6) % 5]
            message = messages[(i // 6) % 3]
            ending = endings[(i // 30) % 2]

            text = f"{problem}. {context}. {message}. {ending}."

            tickets.append({
                "text": text,
                "category": category,
                "priority": priority
            })

df = pd.DataFrame(tickets)

df["text"] = df["text"].str.lower().str.strip()

df = df.drop_duplicates(subset=["text"])

df = df.dropna()

df.to_csv("data/tickets.csv", index=False)

print("DATASET SUMMARY--")
print("Total Tickets:", len(df))

print("\nCategory Distribution:")
print(df["category"].value_counts())

print("\nPriority Distribution:")
print(df["priority"].value_counts())

df["combined"] = df["category"] + "_" + df["priority"]

train, temp = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["combined"]
)

validation, test = train_test_split(
    temp,
    test_size=0.50,
    random_state=42,
    stratify=temp["combined"]
)

train = train.drop(columns=["combined"])
validation = validation.drop(columns=["combined"])
test = test.drop(columns=["combined"])

train.to_csv("data/train.csv", index=False)
validation.to_csv("data/validation.csv", index=False)
test.to_csv("data/test.csv", index=False)

print("\nDATA SPLIT--")
print("Training:", len(train))
print("Validation:", len(validation))
print("Testing:", len(test))