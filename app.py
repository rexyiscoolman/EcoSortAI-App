import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import os

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="EcoSortAI", page_icon="♻️")

# =========================
# CUSTOM CSS FOR COLORS
# =========================
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {background-color: #145214; color: white;}
    [data-testid="stSidebar"] * {color: white !important;}
    .main {background-color: #d0f0c0;}
    .main h1, .main h2, .main h3, .main p, .main span {color: #004d00;}
    .stButton>button {background-color: #4CAF50; color: white; border-radius: 8px; border: none; padding: 0.6em 1.2em;}
    .stButton>button:hover {background-color: #388E3C;}
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# PATHS & CLASSES
# =========================
MODEL_PATH = "recycling_model.keras"
IMG_SIZE = (224, 224)

ALL_CLASSES = [
    "aerosol_cans_default", "aluminum_food_cans_default", "aluminum_soda_cans_default",
    "Battery", "cardboard_boxes_default", "cardboard_packaging_default", "clothing_default",
    "coffee_grounds_default", "diapers", "dirtybags", "disposable_plastic_cutlery_default",
    "eggshells_default", "electronics", "food_waste_default", "glass_beverage_bottles_default",
    "glass_cosmetic_containers_default", "glass_food_jars_default", "Keyboard", "magazines_default",
    "Microwave", "Mobile", "Mouse", "Paper Bag Images", "paperdiapers", "Television",
    "Washing Machine", "augmented"
]
CLASS_COUNT = len(ALL_CLASSES)

# =========================
# MODEL
# =========================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found at `{MODEL_PATH}`.")
        st.stop()
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        st.stop()

model = load_model()

# =========================
# HELPERS
# =========================
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = ImageOps.fit(image, IMG_SIZE, Image.Resampling.LANCZOS)
    arr = np.asarray(image).astype(np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def predict_classes(image: Image.Image):
    x = preprocess_image(image)
    probs = model.predict(x, verbose=0)[0]

    if probs.shape[0] != CLASS_COUNT:
        raise ValueError(f"Model outputs {probs.shape[0]} classes, but ALL_CLASSES has {CLASS_COUNT}.")

    top_idx = int(np.argmax(probs))
    top_prob = float(probs[top_idx])
    predicted_class = ALL_CLASSES[top_idx]

    return predicted_class, top_prob, probs

# =========================
# UI
# =========================
st.sidebar.title("EcoSortAI Menu")
page = st.sidebar.radio("Navigate", ["🏠 Home", "🧠 AI Classifier", "🌍 About Us"])

if page == "🏠 Home":
    st.title("♻️ EcoSortAI: Smart Waste Classifier")
    st.write("Upload a photo of an item and EcoSortAI will classify it into one of **27 categories**.")
    st.info("Tip: Use clear, well-lit photos on a plain background.")

elif page == "🧠 AI Classifier":
    st.title("🧠 Waste Image Classifier")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        CONTAMINATION_CLASSES = [
            "Battery","diapers","dirtybags","electronics","Keyboard",
            "Microwave","Mobile","Mouse","paperdiapers","Television",
            "Washing Machine","augmented"
        ]

        CONTAM_BIAS_MARGIN = 0.05

        predicted_class, _, probs = predict_classes(image)

        # Determine recyclable or contamination
        top_idx = ALL_CLASSES.index(predicted_class)
        top_prob = probs[top_idx]
        max_cont_prob = max([probs[ALL_CLASSES.index(cls)] for cls in CONTAMINATION_CLASSES])

        if predicted_class in CONTAMINATION_CLASSES or max_cont_prob > (top_prob - CONTAM_BIAS_MARGIN):
            result = "Contamination ⚠️"
            st.warning("⚠️ This item is likely contamination and should not go in the recycling bin.")
        else:
            result = "Recyclable ♻️"

        st.subheader(f"Prediction: **{result}**")

elif page == "🌍 About Us":
    st.title("🌍 About EcoSortAI")
    st.write("""
    EcoSortAI builds AI tools that help communities reduce contamination and improve recycling.
    Upload images of waste and get real-time classification into 27 categories.
    """)
