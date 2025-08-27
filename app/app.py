import streamlit as st
import joblib

# Page config
st.set_page_config(page_title="Automated Review Rating System", layout="wide")
st.title("⭐ Automated Review Rating System")

# Load Models from the app folder
try:
    model_a = joblib.load("app/Model_A.pkl")
    model_b = joblib.load("app/Model_B.pkl")
except FileNotFoundError:
    st.error("Model files not found! Make sure Model_A.pkl and Model_B.pkl are in the app folder.")


# User input
review = st.text_area("Enter a review:", height=150)

# Prediction
if st.button("Predict"):
    if review.strip():
        try:
            pred_a = model_a.predict([[review]])[0]  # reshape to 2D
            pred_b = model_b.predict([[review]])[0]


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

