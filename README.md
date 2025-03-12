# **🎬 Movie Review Sentiment Analysis**

🔗 **Live Demo:** [Movie Review Sentiment Analysis App](https://sentiment-analyis-zeeshier.streamlit.app/)

## **📌 Description**  
This project focuses on analyzing **sentiment** in movie reviews using machine learning techniques. The dataset used is the **IMDB Reviews Dataset**, which contains labeled reviews as either positive or negative. The goal is to preprocess text, train machine learning models, and develop an interactive web app for sentiment prediction.

---

## **🚀 Features**  

- 📊 **Data Processing:** Text preprocessing including tokenization, stopword removal, and vectorization.
- 🤖 **Machine Learning Models:** Logistic Regression, Multinomial NB and Random Forest with TF-IDF vectorization.
- 📈 **Model Evaluation:** Accuracy and F1-score used to assess performance.
- 🌐 **Interactive Web App:** Built with Streamlit for user-friendly sentiment analysis.
- 🕙 **Real-Time Predictions:** Enter a movie review and get instant results.

---

## **📂 Dataset**  

- **Kaggle Dataset**: [`IMDB Dataset of 50K Movie Reviews`  ](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/)
- **Includes**: Movie reviews labeled as positive or negative

---

## **🛠 Models Used**  

- ✅ **Logistic Regression**  
- ✅ **Random Forest**  
- ✅ **Multinomial NB**  

---

## **📊 Results & Evaluation**  

- **Accuracy** & **F1-Score** used for performance comparison.
- The model predicts whether a given movie review is **positive** or **negative**. 

---
## **📖 Jupyter Notebook**  

- **File:** movie-review-sentiment-analysis.ipynb
- **Includes:**
  - Data exploration & preprocessing
  - Feature extraction using TF-IDF
  - Model training & evaluation
  - Saving the best accuracy trained model for deployment

Run it with:
```bash
jupyter notebook movie-review-sentiment-analysis.ipynb
``` 
---

## **⚡ Usage**  

1️⃣ **Clone the repository**  
```bash
git clone https://github.com/Zeeshier/Movie-Review-Sentiment-Analysis
cd movie-review-sentiment-analysis
```  

2️⃣ **Install dependencies**  
```bash
pip install -r requirements.txt
```  

3️⃣ **Launch the Streamlit app**  
```bash
streamlit run app.py
```  

---


