import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# ✅ Load dataset safely (raw string avoids path issues)
DATA_PATH = r"C:\Users\Nidhin\OneDrive\Documents\College\Fake News Detector\enhanced_fake_news_dataset.csv"

# Read dataset
df = pd.read_csv(DATA_PATH)

# Display basic info
print("Dataset loaded successfully ✅")
print("Shape:", df.shape)
print("Columns:", df.columns)

# Drop missing values
df = df.dropna()

# Combine title and text columns (modify if your dataset differs)
df["content"] = df["title"] + " " + df["text"]

# Split features and labels
X = df["content"]
y = df["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize text data
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
tfidf_train = vectorizer.fit_transform(X_train)
tfidf_test = vectorizer.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(tfidf_train, y_train)

# Evaluate model
y_pred = model.predict(tfidf_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"\nModel Accuracy: {acc * 100:.2f}%")
print("Confusion Matrix:\n", cm)

# # Save model and vectorizer
# os.makedirs("model", exist_ok=True)
# with open("model/fake_news_model.pkl", "wb") as f:
#     pickle.dump((model, vectorizer), f)

# print("\nModel and vectorizer saved successfully ✅")

print(df['label'].value_counts())
