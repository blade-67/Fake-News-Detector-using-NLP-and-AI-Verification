import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# ✅ Load dataset (modify this path as needed)
DATA_PATH = r"C:\Users\Nidhin\OneDrive\Documents\College\Fake News Detector\enhanced_fake_news_dataset.csv"

# Read dataset
df = pd.read_csv(DATA_PATH)

# Display basic info
print("Dataset loaded successfully ✅")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# Drop missing values
df = df.dropna(subset=["title", "text", "source", "category", "url", "label"])

# ✅ Combine all useful textual columns into one
df["combined"] = (
    df["title"].astype(str) + " " +
    df["text"].astype(str) + " " +
    df["source"].astype(str) + " " +
    df["category"].astype(str) + " " +
    df["url"].astype(str)
)

# Split features and labels
X = df["combined"]
y = df["label"]

# Split dataset into train & test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ✅ TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
tfidf_train = vectorizer.fit_transform(X_train)
tfidf_test = vectorizer.transform(X_test)

# ✅ Train Logistic Regression Model with balanced class weights
model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
model.fit(tfidf_train, y_train)

# ✅ Evaluate Model
y_pred = model.predict(tfidf_test)
y_pred_proba = model.predict_proba(tfidf_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Print detailed metrics
from sklearn.metrics import classification_report
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

print(f"\nModel Accuracy: {acc * 100:.2f}%")
print("Confusion Matrix:\n", cm)

# ✅ Save model and vectorizer
os.makedirs("model", exist_ok=True)
with open("model/fake_news_model.pkl", "wb") as f:
    pickle.dump((model, vectorizer), f)

print("\nModel and vectorizer saved successfully ✅")
print("\nLabel distribution:")
print(df['label'].value_counts())
