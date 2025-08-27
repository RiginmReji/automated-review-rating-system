import streamlit as st
import joblib

# Page config
st.set_page_config(page_title="Automated Review Rating System", layout="wide")
st.title("⭐ Automated Review Rating System — Compare Two Models")

# Load models and vectorizers
try:
    vectorizer_a = joblib.load("app/vectorizer_A.pkl")
    vectorizer_b = joblib.load("app/vectorizer_B.pkl")
    model_a = joblib.load("app/Model_A.pkl")
    model_b = joblib.load("app/Model_B.pkl")
except FileNotFoundError:
    st.error("One or more model/vectorizer files not found! Make sure all .pkl files are in the app folder.")

# User input
review = st.text_area("Enter a review:", height=150)

# Prediction
if st.button("Predict"):
    if review.strip():
        try:
            # Transform text using vectorizers
            review_a = vectorizer_a.transform([review])
            review_b = vectorizer_b.transform([review])

            # Predict
            pred_a = model_a.predict(review_a)[0]
            pred_b = model_b.predict(review_b)[0]

            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Model A (Balanced)")
                st.write("Prediction:", pred_a)
            with col2:
                st.subheader("Model B (Imbalanced)")
                st.write("Prediction:", pred_b)

        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.warning("Please enter a review before predicting.")


