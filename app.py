import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download stopwords
nltk.download('stopwords')
nltk.download('punkt')

# Load model and vectorizer
with open("logistic_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)  
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

def predict_sentiment(review):
    processed_review = preprocess_text(review)
    vectorized_review = vectorizer.transform([processed_review])
    prediction = model.predict(vectorized_review)[0]
    return '😊 Positive' if prediction == 1 else '😞 Negative'

# Streamlit UI
st.set_page_config(page_title="Sentiment Analysis", page_icon="🎬", layout="centered")

# Custom CSS for styling
st.markdown(
    """
    <style>
        .stApp {
            background-color: #f0f2f6;
        }
        .title {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            color: #1f77b4;
        }
        .text-input {
            border-radius: 10px;
            padding: 10px;
        }
        .btn {
            background-color: #1f77b4;
            color: white;
            border-radius: 10px;
            padding: 10px;
            font-size: 16px;
            width: 100%;
            text-align: center;
        }
        .result {
            font-size: 24px;
            font-weight: bold;
            text-align: center;
            margin-top: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<h1 class="title">🎬 Movie Review Sentiment Analysis 🎭</h1>', unsafe_allow_html=True)
st.markdown("#### Enter a movie review below to analyze its sentiment.")

# User input text box
user_input = st.text_area("Write your review here...", height=150, key="input_review")

# Predict Button
if st.button("Analyze Sentiment", help="Click to analyze the sentiment of your review"):
    result = predict_sentiment(user_input)
    st.markdown(f'<div class="result">{result}</div>', unsafe_allow_html=True)
