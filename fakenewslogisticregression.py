import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Setup
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Text cleaning function
def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text.lower())
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Load data
@st.cache_data
def load_data():
    true_df = pd.read_csv('https://raw.githubusercontent.com/raj4angular/fakenewsstreamlit/main/True_Sample.csv')
    fake_df = pd.read_csv('https://raw.githubusercontent.com/raj4angular/fakenewsstreamlit/main/Fake_Sample.csv')
    true_df['label'] = 0
    fake_df['label'] = 1
    df = pd.concat([true_df, fake_df]).sample(frac=1).reset_index(drop=True)
    df['text'] = df['title'] + " " + df['text']
    df.drop(['title', 'subject', 'date'], axis=1, inplace=True)
    df['cleaned_text'] = df['text'].apply(clean_text)
    return df

# Train model
@st.cache_resource
def train_model(df):
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X = vectorizer.fit_transform(df['cleaned_text'])
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(class_weight='balanced')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]
    metrics = {
        "report": classification_report(y_test, y_pred, output_dict=True),
        "confusion": confusion_matrix(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_probs),
        "vectorizer": vectorizer,
        "model": model
    }
    return metrics

# Streamlit UI
st.title("📰 Fake News Detector")
st.markdown("Detect whether a news article is real or fake using Logistic Regression.")

df = load_data()
metrics = train_model(df)

# Prediction section
st.header("🔍 Try it out")
user_input = st.text_area("Enter news text here:")
if st.button("Predict"):
    cleaned = clean_text(user_input)
    vectorized = metrics["vectorizer"].transform([cleaned])
    prediction = metrics["model"].predict(vectorized)[0]
    prob = metrics["model"].predict_proba(vectorized)[0][prediction]
    label = "Real News" if prediction == 0 else "Fake News"
    st.success(f"Prediction: {label} ({prob:.2f} confidence)")

# Metrics section
st.header("📊 Model Performance")
st.write("ROC-AUC Score:", round(metrics["roc_auc"], 3))
st.subheader("Confusion Matrix")
st.dataframe(pd.DataFrame(metrics["confusion"], columns=["Predicted Real", "Predicted Fake"], index=["Actual Real", "Actual Fake"]))
st.subheader("Classification Report")
st.json(metrics["report"])

# Optional: show data
with st.expander("📁 Show Sample Data"):
    st.dataframe(df[['text', 'label']].head(10))
