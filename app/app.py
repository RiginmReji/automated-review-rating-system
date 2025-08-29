import re
import streamlit as st
import joblib
import nltk
from nltk.corpus import words

# -------------------------------
# Download English words list
# -------------------------------
nltk.download('words')
english_vocab = set(words.words())

# -------------------------------
# Streamlit page configuration
# -------------------------------
st.set_page_config(page_title="Automated Review Rating System", layout="wide")
st.title("⭐ Automated Review Rating System — Compare Two Models")

# -------------------------------
# Load pipelines
# -------------------------------
try:
    model_a = joblib.load("app/Model_A_pipeline.pkl")  # Balanced pipeline
    model_b = joblib.load("app/Model_B_pipeline.pkl")  # Imbalanced pipeline
except Exception as e:
    st.error("❌ Model files not found! Make sure both pipeline .pkl files are in the 'app' folder.")
    st.stop()

# -------------------------------
# Cleaning function
# -------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # remove numb







