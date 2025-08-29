import re
import streamlit as st
import joblib

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
    text = re.sub(r'[^a-z\s]', '', text)  # remove numbers and symbols
    return text.strip()

# -------------------------------
# User input
# -------------------------------
review = st.text_area("✍️ Enter a review:", height=150)

# -------------------------------
# Prediction button
# -------------------------------
if st.button("Predict"):
    cleaned_review = clean_text(review)

    # Prevent prediction if input is empty after cleaning
    if not cleaned_review:
        st.warning("⚠️ Please enter valid text (letters only). Numbers and symbols are ignored.")
    else:
        try:
            # Predict using both models
            pred_a = model_a.predict([cleaned_review])[0]
            pred_b = model_b.predict([cleaned_review])[0]

            # Display side by side
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("✅ Model A (Balanced)")
                st.write(f"Prediction: ⭐ {pred_a}")

            with col2:
                st.subheader("⚖️ Model B (Imbalanced)")
                st.write(f"Prediction: ⭐ {pred_b}")

        except Exception as e:
            st.error(f"⚠️ Error during prediction: {e}")






