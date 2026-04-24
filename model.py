import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("resume_dataset.csv")

# Clean text
def clean_text(text):
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

df["Resume"] = df["Resume"].apply(clean_text)

# Convert text to numbers
tfidf = TfidfVectorizer(stop_words="english")

X = tfidf.fit_transform(df["Resume"])
y = df["Category"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()

model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

# Test sample resume
sample_resume = """
Python Machine Learning Data Analysis NLP Deep Learning
"""

sample_resume = clean_text(sample_resume)

vector = tfidf.transform([sample_resume])

prediction = model.predict(vector)

print("Predicted Category:", prediction[0])