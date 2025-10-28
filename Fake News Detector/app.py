# ==========================================
# app.py — Streamlit Fake News Detector
# ==========================================

import streamlit as st
import pickle
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai

# ------------------------------------------
# 1️⃣ Load trained model and vectorizer
# ------------------------------------------
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "fake_news_model.pkl")
with open(MODEL_PATH, "rb") as f:
    model, vectorizer = pickle.load(f)

# ------------------------------------------
# 2️⃣ Configure Gemini API (Optional)
# ------------------------------------------
# ⚠️ Replace "YOUR_GEMINI_API_KEY" with your actual key if you want this feature
genai.configure(api_key="AIzaSyDdjfiHQNnpLJ1Yf7JMyrpIwuw_GBBn1Gc")

# ------------------------------------------
# 3️⃣ Helper: Extract article text from URL
# ------------------------------------------
def extract_article_text(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")

        # Extract paragraphs
        paragraphs = [p.get_text() for p in soup.find_all("p")]
        article = " ".join(paragraphs)
        return article if len(article) > 200 else None
    except Exception as e:
        st.error(f"Error extracting article: {e}")
        return None

# ------------------------------------------
# 4️⃣ Helper: Gemini Verification (optional)
# ------------------------------------------
def gemini_check(content):
    try:
        prompt = f"Analyze this news content and return only 'REAL' or 'FAKE':\n\n{content[:1000]}"
        model_g = genai.GenerativeModel("gemini-pro")
        response = model_g.generate_content(prompt)
        verdict = response.text.strip().upper()
        return "REAL" if "REAL" in verdict else "FAKE"
    except Exception as e:
        st.error(f"Gemini API error: {str(e)}")
        return "UNKNOWN"

# ------------------------------------------
# 5️⃣ Streamlit UI
# ------------------------------------------
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.title("📰 Fake News Detection using NLP")
st.write("Check if a news article or URL is real or fake using a trained ML model + optional Gemini verification.")

# Option selector
option = st.radio("Choose Input Type:", ["Enter Text", "Enter Article URL"])

if option == "Enter Text":
    user_text = st.text_area("Paste the news article text below 👇", height=200)

    if st.button("Classify"):
        if user_text.strip():
            # Transform and predict
            X_input = vectorizer.transform([user_text])
            prediction = model.predict(X_input)[0]

            st.subheader("🧠 Model Prediction:")
            st.write(f"**This news seems to be:** {prediction.upper()}")

            # Optional Gemini verification (edge case check)
            if st.checkbox("Verify with Gemini AI (Optional)"):
                verdict = gemini_check(user_text)
                st.write(f"🤖 Gemini’s Verdict: **{verdict}**")
        else:
            st.warning("Please enter some text!")

elif option == "Enter Article URL":
    user_url = st.text_input("Paste the news article URL below 👇")

    if st.button("Fetch & Classify"):
        if user_url.strip():
            article = extract_article_text(user_url)
            if article:
                st.info("✅ Article extracted successfully!")
                X_input = vectorizer.transform([article])
                prediction = model.predict(X_input)[0]

                st.subheader("🧠 Model Prediction:")
                st.write(f"**This news seems to be:** {prediction.upper()}")

                if st.checkbox("Verify with Gemini AI (Optional)"):
                    verdict = gemini_check(article)
                    st.write(f"🤖 Gemini’s Verdict: **{verdict}**")
            else:
                st.error("Could not extract article text. Try a different URL.")
        else:
            st.warning("Please enter a valid URL!")

 
# 6️⃣ Footer
st.markdown("---")
st.caption("Developed by Nidhin — Fake News Detection using NLP & Gemini AI")
