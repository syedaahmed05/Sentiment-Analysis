import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
# Load the dataset
df = pd.read_csv("amazon_reviews.csv")
df = df.dropna(subset=['reviewText'])
# Label sentiment
df["sentiment"] = df["overall"].apply(lambda x: "Negative" if x <= 2 else
"Neutral" if x == 3 else "Positive")
# Show sentiment distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="sentiment", order=["Negative", "Neutral",
"Positive"])
plt.title("Number of Reviews by Sentiment")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
# Clean text
def clean_text(text):
text = text.lower()
text = re.sub(r'\W', ' ', text)
text = re.sub(r'\s+', ' ', text).strip()
return text
df["cleaned_review"] = df["reviewText"].apply(clean_text)
# Split data
X_train, X_test, y_train, y_test = train_test_split(
df["cleaned_review"], df["sentiment"],
test_size=0.2, stratify=df["sentiment"], random_state=42
)
# Vectorize text
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train_tfidf, y_train)
# Predict and evaluate
y_pred = model.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))
# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=["Negative", "Neutral",
"Positive"])
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
xticklabels=["Negative", "Neutral", "Positive"],
yticklabels=["Negative", "Neutral", "Positive"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
