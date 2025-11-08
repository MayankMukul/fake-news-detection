import streamlit as st
import joblib

st.title("📰 Fake News Detection System")
st.write("Enter a news article below to classify it as Fake or Real.")

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# User input
news = st.text_area("Enter news text here...")

if st.button("Predict"):
    if news.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Transform input
        data = vectorizer.transform([news])
        result = model.predict(data)[0]

        if result == 1:
            st.success("✅ REAL NEWS")
        else:
            st.error("❌ FAKE NEWS")
