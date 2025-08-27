import streamlit as st
import joblib

st.set_page_config(page_title="Automated Review Rating System", layout="wide")
st.title("⭐ Automated Review Rating System ")

# Load models
model_a = joblib.load("Model_A.pkl")
model_b = joblib.load("Model_B.pkl")

review = st.text_area("Enter a review:", height=150)

if st.button("Predict"):
    if review.strip():
        pred_a = model_a.predict([review])[0]
        pred_b = model_b.predict([review])[0]

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Model A (Balanced)")
            st.write("Prediction:", pred_a)

        with col2:
            st.subheader("Model B (Imbalanced)")
            st.write("Prediction:", pred_b)
    else:
        st.warning("Please enter a review before predicting.")
