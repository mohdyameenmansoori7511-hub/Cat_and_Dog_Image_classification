"""
===============================================================================
Project ID   : DL-003
Project Title: Cat and Dog Image Classification - Interactive Web App
Group ID     : BTech_DL_041 | Team Leader: Shivam Chauhan (BCS2024016)
Mentor Name  : Mr. Pradeep Singh | Invertis University Bareilly (CSED)
Deployment   : Streamlit Community Cloud / Hugging Face Spaces
===============================================================================
"""

from pathlib import Path
from PIL import Image
import numpy as np
import streamlit as st
import tensorflow as tf

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="DL-003: Cat vs. Dog Classifier",
    page_icon="🐾",
    layout="wide",
)

IMG_SIZE = (160, 160)
CLASS_NAMES = ["Cat 🐱", "Dog 🐶"]

# -----------------------------------------------------------------------------
# 2. Model Loader (Cached for Fast Web Hosting)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_classifier_model():
    """
    Loads `cat_dog_classifier.keras` if uploaded in the repository.
    If the .keras file is not found, builds a lightweight MobileNetV2-based
    Cat vs. Dog detector so the hosted demo works out-of-the-box on the web.
    """
    candidate_paths = [
        Path("cat_dog_classifier.keras"),
        Path("dl003_cats_dogs_workspace/outputs/cat_dog_classifier.keras"),
    ]
    for path in candidate_paths:
        if path.exists():
            model = tf.keras.models.load_model(path)
            return model, "Custom Trained Compact CNN (`cat_dog_classifier.keras`)"

    # Fallback: Pre-trained MobileNetV2 (ImageNet classes 151-268 = Dogs, 281-285 = Cats)
    base_model = tf.keras.applications.MobileNetV2(weights="imagenet")
    return base_model, "Pre-Trained MobileNetV2 Fallback (Upload `cat_dog_classifier.keras` for custom CNN)"


def predict_pet(image: Image.Image, model, model_type: str) -> tuple[str, float, float, float]:
    """Returns (predicted_label, confidence_pct, cat_prob_pct, dog_prob_pct)."""
    if "Custom Trained" in model_type:
        img_resized = image.convert("RGB").resize(IMG_SIZE)
        arr = np.expand_dims(np.array(img_resized, dtype="float32"), axis=0)
        dog_prob = float(model.predict(arr, verbose=0)[0][0])
        cat_prob = 1.0 - dog_prob
    else:
        # MobileNetV2 ImageNet 224x224 inference aggregated across Cat & Dog synsets
        img_resized = image.convert("RGB").resize((224, 224))
        arr = np.expand_dims(np.array(img_resized, dtype="float32"), axis=0)
        arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
        preds = model.predict(arr, verbose=0)[0]
        # ImageNet indices: Domestic Cats (281..285), Domestic Dogs (151..268)
        cat_score = float(np.sum(preds[281:286]))
        dog_score = float(np.sum(preds[151:269]))
        total = cat_score + dog_score + 1e-8
        dog_prob = dog_score / total
        cat_prob = cat_score / total

    if dog_prob >= 0.5:
        return CLASS_NAMES[1], dog_prob * 100.0, cat_prob * 100.0, dog_prob * 100.0
    return CLASS_NAMES[0], cat_prob * 100.0, cat_prob * 100.0, dog_prob * 100.0


# -----------------------------------------------------------------------------
# 3. Sidebar: Project Metadata & Team Details
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎓 Invertis University Bareilly")
    st.caption("**Center for Skill & Entrepreneurship Development (CSED)**")
    st.markdown(
        """
        - **Project ID:** `DL-003`
        - **Group ID:** `BTech_DL_041` (Sem V, Sec E)
        - **Domain:** Computer Vision / Binary Classification
        - **Mentor:** Mr. Pradeep Singh
        - **Team Leader:** **Shivam Chauhan** (`BCS2024016`)
        """
    )
    st.markdown("### 📊 Colab Benchmark Metrics")
    st.metric("Dataset Size", "3,000 Images", "70% Train / 15% Val / 15% Test")
    st.metric("Unseen Test Accuracy", "69.11% (CNN)", "96.00% with MobileNetV2")
    st.metric("Dog Precision / Cat Recall", "75.00% / 80.89%", "450 Unseen Test Images")

# -----------------------------------------------------------------------------
# 4. Main Web App Interface
# -----------------------------------------------------------------------------
st.title("🐾 Cat and Dog Image Classification (DL-003)")
st.markdown(
    "Upload any photo of a **Cat** or **Dog** (indoor, outdoor grass, sofa, or street background) "
    "to classify it in real time using our trained Convolutional Neural Network."
)

model, model_status = load_classifier_model()
st.info(f"**Active Model Engine:** {model_status}")

col_upload, col_result = st.columns([1, 1], gap="large")

with col_upload:
    st.subheader("1️⃣ Upload Pet Image")
    uploaded_file = st.file_uploader(
        "Choose a JPG or PNG image...",
        type=["jpg", "jpeg", "png", "webp"],
    )
    if uploaded_file is not None:
        user_img = Image.open(uploaded_file)
        st.image(user_img, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)

with col_result:
    st.subheader("2️⃣ Classification Result")
    if uploaded_file is not None:
        with st.spinner("Running CNN inference on image..."):
            pred_label, conf_pct, cat_pct, dog_pct = predict_pet(user_img, model, model_status)

        st.success(f"### Predicted Class: **{pred_label}**")
        st.metric(label="Prediction Confidence", value=f"{conf_pct:.2f}%")

        st.markdown("#### Class Probability Breakdown")
        st.write(f"🐱 **Cat Probability:** `{cat_pct:.2f}%`")
        st.progress(int(round(cat_pct)))
        st.write(f"🐶 **Dog Probability:** `{dog_pct:.2f}%`")
        st.progress(int(round(dog_pct)))
    else:
        st.write("👈 Upload an image on the left to see the prediction and confidence scores.")

st.divider()
with st.expander("👥 View Full Team Roster (Group ID: BTech_DL_041 - Section E)"):
    st.table(
        [
            {"Sr. No.": 1, "Student Name": "Shivam Chauhan (Team Leader)", "Student ID": "BCS2024016", "Roll No.": "2410303464"},
            {"Sr. No.": 2, "Student Name": "Mohd Adnan", "Student ID": "BCS2024007", "Roll No.": "2410303271"},
            {"Sr. No.": 3, "Student Name": "Krish Gupta", "Student ID": "BCS2024013", "Roll No.": "2410303234"},
            {"Sr. No.": 4, "Student Name": "Ginesh Singh", "Student ID": "BCS2024020", "Roll No.": "2410303186"},
            {"Sr. No.": 5, "Student Name": "Neil Bhardwaj", "Student ID": "BCS2024030", "Roll No.": "2410303306"},
            {"Sr. No.": 6, "Student Name": "Hirdesh Kumar", "Student ID": "BCS2024033", "Roll No.": "2410303209"},
            {"Sr. No.": 7, "Student Name": "Mohit Babu", "Student ID": "BCS2024034", "Roll No.": "2410303293"},
            {"Sr. No.": 8, "Student Name": "Shivam Sharma", "Student ID": "BCS2024035", "Roll No.": "2410303467"},
            {"Sr. No.": 9, "Student Name": "Ramjeet Yadav", "Student ID": "BCS2024039", "Roll No.": "2410303385"},
            {"Sr. No.": 10, "Student Name": "Shaurya Kumar Yadav", "Student ID": "BCS2024049", "Roll No.": "2410303452"},
            {"Sr. No.": 11, "Student Name": "Shivanshu Sharma", "Student ID": "BCS2024056", "Roll No.": "2410303473"},
            {"Sr. No.": 12, "Student Name": "Mohd Yameen", "Student ID": "BCS2024066", "Roll No.": "2410303289"},
            {"Sr. No.": 13, "Student Name": "Akhilesh Bhargav", "Student ID": "BCS2024081", "Roll No.": "2410303048"},
            {"Sr. No.": 14, "Student Name": "Siddharth Mishra", "Student ID": "BCS2024087", "Roll No.": "2410303481"},
        ]
    )