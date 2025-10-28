import streamlit as st
import pickle
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
import os

# Load model and configure API
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "fake_news_model.pkl")
with open(MODEL_PATH, "rb") as f:
    model, vectorizer = pickle.load(f)

api_key = "API_KEY"
genai.configure(api_key=api_key)

def analyze_text(text):
    """Analyze text using ML model and return prediction with confidence"""
    X_input = vectorizer.transform([text])
    prediction = model.predict(X_input)[0]
    prediction_proba = model.predict_proba(X_input)[0]
    confidence = prediction_proba.max() * 100
    return prediction, confidence

def extract_article_text(url):
    """Extract article text from URL"""
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")
        paragraphs = [p.get_text() for p in soup.find_all("p")]
        article = " ".join(paragraphs)
        return article if len(article) > 200 else None
    except Exception as e:
        st.error(f"Error extracting article: {e}")
        return None

def gemini_check(content):
    """Get Gemini AI's verification of the content"""
    try:
        prompt = f"Analyze this content and determine if it's FAKE or REAL news. Explain why in one sentence.\n\n{content[:1000]}"
        model_g = genai.GenerativeModel("gemini-pro")
        response = model_g.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Gemini analysis unavailable: {str(e)}"

# Page setup
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")
st.title("📰 Fake News Detector")
st.write("Detect fake or real news using ML and optional Gemini AI verification")

# Input selection
option = st.radio("Choose Input Type:", ["Enter Text", "Enter Article URL"])

if option == "Enter Text":
    text_input = st.text_area("Paste the news article text:", height=200)
    
    if st.button("Analyze") and text_input.strip():
        # Display truncated input
        st.markdown("### 📝 Input Text")
        st.write(text_input[:300] + ("..." if len(text_input) > 300 else ""))
        
        # ML Model prediction
        prediction, confidence = analyze_text(text_input)
        st.markdown("### � ML Model Analysis")
        
        if prediction.lower() == 'fake':
            st.error(f"Verdict: **FAKE NEWS** (Confidence: {confidence:.1f}%)")
        else:
            st.success(f"Verdict: **REAL NEWS** (Confidence: {confidence:.1f}%)")
        
        if confidence < 70:
            st.warning("⚠️ Low confidence - please verify with additional sources")
        
        # Optional Gemini verification
        if st.checkbox("Get Second Opinion (Gemini AI)"):
            with st.spinner("Analyzing with Gemini AI..."):
                gemini_result = gemini_check(text_input)
            st.markdown("### 🔍 Gemini AI Analysis")
            st.write(gemini_result)

elif option == "Enter Article URL":
    url_input = st.text_input("Enter article URL:")
    
    if st.button("Analyze") and url_input.strip():
        with st.spinner("Fetching article..."):
            article_text = extract_article_text(url_input)
            
        if article_text:
            # ML Model prediction
            prediction, confidence = analyze_text(article_text)
            st.markdown("### � ML Model Analysis")
            
            if prediction.lower() == 'fake':
                st.error(f"Verdict: **FAKE NEWS** (Confidence: {confidence:.1f}%)")
            else:
                st.success(f"Verdict: **REAL NEWS** (Confidence: {confidence:.1f}%)")
            
            if confidence < 70:
                st.warning("⚠️ Low confidence - please verify with additional sources")
            
            # Optional Gemini verification
            if st.checkbox("Get Second Opinion (Gemini AI)"):
                with st.spinner("Analyzing with Gemini AI..."):
                    gemini_result = gemini_check(article_text)
                st.markdown("### 🔍 Gemini AI Analysis")
                st.write(gemini_result)
        else:
            st.error("Could not extract article text. Please try a different URL.")

st.markdown("---")
st.caption("Developed by Nidhin | 2025")

