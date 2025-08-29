import streamlit as st
import joblib

st.set_page_config(page_title="Automated Review Rating System", layout="wide")
st.title("⭐ Automated Review Rating System — Compare Two Models")

# Try loading the models
try:
    model_a = joblib.load("app/Model_A_pipeline(1).pkl")  # Balanced pipeline
    model_b = joblib.load("app/Model_B_pipeline.pkl")  # Imbalanced pipeline
except Exception as e:
    st.error("❌ Model files not found! Please make sure Model_A_pipeline.pkl and Model_B_pipeline.pkl are inside the 'app' folder.")
    st.stop()

# Input
review = st.text_area("✍️ Enter a product review:", height=150)

if st.button("Predict"):
    if review.strip():
        try:
            # Predictions
            pred_a = model_a.predict([review])[0]
            pred_b = model_b.predict([review])[0]

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("✅ Model A (Balanced)")
                st.write(f"Prediction: ⭐ {pred_a}")

            with col2:
                st.subheader("⚖️ Model B (Imbalanced)")
                st.write(f"Prediction: ⭐ {pred_b}")

        except Exception as e:
            st.error(f"⚠️ Error during prediction: {e}")
    else:
        st.warning("⚠️ Please enter a review before predicting.")




