# 🧠 Fake News Detection Model (with Gemini Verification & Web Scraping)

## 📘 Overview
This project detects fake news using Machine Learning and real-world article data.  
It combines text classification, web scraping, and optional AI verification to ensure accurate and trustworthy results.  
Users can input a news article or a web URL directly through an interactive **Streamlit dashboard**, making it easy to analyze and verify news authenticity.


## 🚀 Key Features
- 📰 **Fake News Detection** using Logistic Regression & Passive Aggression Classifier  
- 🌐 **Web Scraping** with BeautifulSoup to extract article content from URLs  
- 🤖 **Optional Gemini API Verification** for additional factual cross-checking  
- 📊 **Streamlit Dashboard** for real-time user interaction and visualization  
- 🔍 Supports multiple input sources — title, text, or live URL  


## 🧩 Tech Stack
| Component | Technology |
|------------|-------------|
| Language | Python |
| Web Framework | Streamlit |
| Machine Learning | Logistic Regression, PassiveAggressionClassifier |
| Web Scraping | BeautifulSoup |
| API Integration | Gemini API |
| Libraries | Pandas, Scikit-learn, Numpy, Requests |


## ⚙️ Workflow
1. **User Input:** Enter a news article or paste a web URL.  
2. **Scraping & Preprocessing:** Text extracted using BeautifulSoup and cleaned for model prediction.  
3. **Prediction:** ML model classifies the input as *Real* or *Fake*.  
4. **Optional Verification:** Gemini API provides factual comparison if the user chooses.  
5. **Display Results:** Results shown interactively on the Streamlit dashboard.
