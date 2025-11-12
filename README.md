📰 Fake News Detection – Machine Learning Project
📌 Overview

This project is a complete Fake News Detection System built using Machine Learning and Natural Language Processing (NLP).
It automatically classifies news articles as Real or Fake using trained ML models and text-processing techniques.

The project includes:

Dataset preprocessing

Model training

Evaluation and metrics

Saving the trained model

A deployment-ready prediction script

Streamlit GUI for testing news in real time

🚀 Features
✔ Fake vs Real News Classification

Uses ML algorithms trained on labeled news datasets.

✔ End-to-End Pipeline

Load and combine datasets

Clean and preprocess text

Convert text to numerical vectors (TF-IDF)

Train a machine learning classifier

Save the model using joblib

Predict user-given news statements

✔ Streamlit Web App

A user-friendly interface to enter news content and get predictions instantly.

✔ Deployment Support

Project structure and model files are ready for deployment on platforms like Render, HuggingFace Spaces, or local machines.

📂 Project Structure
fake-news-detection/
│
├── dataset/
│    ├── Fake.csv
│    ├── True.csv
│
├── model.joblib
├── vectorizer.joblib
│
├── fake_news_project.py   # Main ML training & prediction script
├── app.py                 # Streamlit GUI
│
└── README.md

🧠 Technologies Used
Component	Technology
Programming Language	Python
Machine Learning	Logistic Regression / Passive Aggressive Classifier
Text Processing	NLTK, Scikit-learn, TF-IDF
Deployment	Streamlit
Model Saving	Joblib
🗂 Dataset

You must provide two CSV files inside the dataset folder:

Fake.csv – Contains fake news articles

True.csv – Contains real news articles

Both contain columns like:

title

text

subject

date

These two are merged and labelled for training.

🧹 Data Preprocessing

The system performs:

Lowercasing

Removing punctuation & stopwords

Tokenization

Lemmatization (optional)

Label encoding

Train-test split

TF-IDF vectorization

🤖 Model Training

The following steps are performed:

Load dataset

Clean the text

Convert text into TF-IDF vectors

Train the ML classifier

Evaluate the model on test data

Save the trained model using joblib

Example metrics (depends on dataset):

Accuracy: ~94%

Precision & Recall: High for both classes

💾 Saved Model

Two important files are generated:

File	Purpose
model.joblib	Stores the trained ML classifier
vectorizer.joblib	Stores the TF-IDF vectorizer

These are used during prediction and deployment.

🖥 Running the Model Locally
1️⃣ Install required dependencies
pip install -r requirements.txt

2️⃣ Train the model (if needed)
python fake_news_project.py

3️⃣ Run the Streamlit Web App
streamlit run app.py

🌐 Deployment

This project can be deployed on:

Render

HuggingFace Spaces

Streamlit Cloud

Railway

Localhost

Ensure you upload:

model.joblib

vectorizer.joblib

app.py

requirements.txt

🧪 Example Prediction

Input:

“Government announces new policy related to fuel prices.”

Output:

Real News

Input:

“NASA confirms aliens have arrived in India!”

Output:

Fake News

📘 Future Improvements

Use Deep Learning (LSTMs, BERT, DistilBERT)

Add explainability (LIME, SHAP)

Improve UI with better dashboard

Deploy model as a REST API (FastAPI / Flask)

Add multilingual fake news detection

🤝 Contribution

Feel free to fork the repository, raise issues, or submit PRs to improve the project.

📜 License

This project is open-source under the MIT License.