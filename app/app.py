import streamlit as st
import joblib
import re

# -------------------------------
# Helper: clean text same as training
# -------------------------------
def clean_text(text):
    text = text.lower()  
    text = re.sub(r'[^a-z\s]', '', text)  # keep only letters and spaces
    return text.strip()

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Automated Review Rating System", layout="wide")
st.title("⭐ Automated Review Rating System — Compare Two Models")

# -------------------------------
# Load Models + Vectorizer
# -------------------------------
try:
    model_a = joblib.load("app/Model_A.pkl")
    model_b = joblib.load("app/Model_B.pkl")
    vectorizer = joblib.load("app/vectorizer_A.pkl")   # ✅ load vectorizer
except Exception as e:
    st.error(f"One or more model/vectorizer files not found: {e}")
    st.stop()

# -------------------------------
# Input Review
# -------------------------------
review = st.text_area("Enter a review:", height=150)

if st.button("Predict"):
    if review.strip():
        try:
            # Clean + transform
            cleaned_review = clean_text(review)
            review_vectorized = vectorizer.transform([cleaned_review])

            # Predictions
            pred_a = model_a.predict(review_vectorized)[0]
            pred_b = model_b.predict(review_vectorized)[0]

            # Display Results
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



