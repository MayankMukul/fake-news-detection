import streamlit as st
import joblib

st.title("📰 Fake News Detection System (Deployment Template)")
st.write("This is a deploy-ready template. Replace model.pkl and vectorizer.pkl with trained files.")

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

news = st.text_area("Enter news text...")

if st.button("Predict"):
    vec = vectorizer.transform([news])
    pred = model.predict(vec)[0]
    st.success("Real News ✅" if pred==1 else "Fake News ❌")
