import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import keras

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
    [data-testid="stSidebar"] {
        background-color: #145214;
        color: white;
    }

    [data-testid="stSidebar"] * {
        color: white !important;
    }

    .main {
        background-color: #d0f0c0;
    }

    .main h1, .main h2, .main h3, .main p, .main span {
        color: #004d00;
    }

    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.6em 1.2em;
    }

    .stButton>button:hover {
        background-color: #388E3C;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# PATHS & CLASSES
# =========================
MODEL_PATH = "/Users/shanyusuraparaju/Documents/Recycling_Images/recycling_model"
IMG_SIZE = (224, 224)

ALL_CLASSES = [
    "aerosol_cans_default",
    "aluminum_food_cans_default",
    "aluminum_soda_cans_default",
    "Battery",
    "cardboard_boxes_default",
    "cardboard_packaging_default",
    "clothing_default",
    "coffee_grounds_default",
    "diapers",
    "dirtybags",
    "disposable_plastic_cutlery_default",
    "eggshells_default",
    "electronics",
    "food_waste_default",
    "glass_beverage_bottles_default",
    "glass_cosmetic_containers_default",
    "glass_food_jars_default",
    "Keyboard",
    "magazines_default",
    "Microwave",
    "Mobile",
    "Mouse",
    "Paper Bag Images",
    "paperdiapers",
    "Television",
    "Washing Machine",
    "augmented"
]
CLASS_COUNT = len(ALL_CLASSES)

# =========================
# MODEL (Keras 3 fix)
# =========================
@st.cache_resource
def load_model():
    try:
        # Try loading as a new .keras or .h5 model first
        return tf.keras.models.load_model(MODEL_PATH, compile=False)
    except ValueError:
        # Fallback for Keras 3 SavedModel directories
        st.warning("⚠️ Using Keras 3 TFSMLayer loader for SavedModel format.")
        return keras.layers.TFSMLayer(MODEL_PATH, call_endpoint="serving_default")

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
    probs = model(x, training=False)[0].numpy()

    if probs.shape[0] != CLASS_COUNT:
        raise ValueError(
            f"Model outputs {probs.shape[0]} classes, but ALL_CLASSES has {CLASS_COUNT}."
        )

    top_idx = int(np.argmax(probs))
    top_prob = float(probs[top_idx])
    predicted_class = ALL_CLASSES[top_idx]

    return predicted_class, top_prob, probs

# =========================
# UI: SIDEBAR NAV
# =========================
st.sidebar.title("EcoSortAI Menu")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "🧠 AI Classifier", "🌍 About Us", "💳 Purchase Info", "📝 Feedback"]
)

# =========================
# PAGES
# =========================
if page == "🏠 Home":
    st.title("♻️ EcoSortAI: Smart Waste Classifier")
    st.write(
        "Upload a photo of an item and EcoSortAI will classify it into one of **27 categories**. "
        "This helps communities better understand recycling and contamination streams."
    )
    st.info("Tip: For best results, use clear, well-lit photos on a plain background.")

elif page == "🧠 AI Classifier":
    st.title("🧠 Waste Image Classifier")
    uploaded_file = st.file_uploader("Upload an image of waste", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        RECYCLABLE_CLASSES = [
            "aerosol_cans_default",
            "aluminum_food_cans_default",
            "aluminum_soda_cans_default",
            "cardboard_boxes_default",
            "cardboard_packaging_default",
            "clothing_default",
            "coffee_grounds_default",
            "disposable_plastic_cutlery_default",
            "eggshells_default",
            "glass_beverage_bottles_default",
            "glass_cosmetic_containers_default",
            "glass_food_jars_default",
            "magazines_default",
            "Paper Bag Images",
        ]

        CONTAMINATION_CLASSES = [
            "Battery",
            "diapers",
            "dirtybags",
            "electronics",
            "Keyboard",
            "Microwave",
            "Mobile",
            "Mouse",
            "paperdiapers",
            "Television",
            "Washing Machine",
            "augmented"
        ]

        CONTAM_BIAS_MARGIN = 0.05

        def classify_item(predicted_class: str, probs: np.ndarray) -> str:
            top_idx = ALL_CLASSES.index(predicted_class)
            top_prob = probs[top_idx]
            cont_probs = [probs[ALL_CLASSES.index(cls)] for cls in CONTAMINATION_CLASSES]
            max_cont_prob = max(cont_probs)

            if predicted_class in CONTAMINATION_CLASSES or max_cont_prob > (top_prob - CONTAM_BIAS_MARGIN):
                return "Contamination ⚠️"
            else:
                return "Recyclable ♻️"

        try:
            predicted_class, _, probs = predict_classes(image)
        except Exception as e:
            st.error(str(e))
        else:
            result = classify_item(predicted_class, probs)
            st.subheader(f"Prediction: **{result}**")
            if result == "Contamination ⚠️":
                st.warning("⚠️ This item is likely **contamination** and should not go in the recycling bin.")

elif page == "🌍 About Us":
    st.title("🌍 About EcoSortAI")
    st.write("""
### Who We Are
EcoSortAI builds AI tools that help communities and businesses **reduce contamination** and **improve recycling**.

### What We Do
- Real-time image classification of common waste items  
- Education and guidance to reduce recycling errors  
- Tools that scale from individuals to enterprises

### Our Mission
Make correct sorting **easy**, **fast**, and **accessible** for everyone.
    """)

elif page == "💳 Purchase Info":
    st.title("💳 Subscription Plans")
    st.write("Pick a plan that fits your needs:")

    plans = {
        "Free": {
            "Price": "$0 / month",
            "Features": [
                "Upload up to 20 images/month",
                "Basic AI classification",
                "Community support"
            ]
        },
        "Pro": {
            "Price": "$15 / month",
            "Features": [
                "Unlimited image uploads",
                "Priority model updates",
                "Email support"
            ]
        },
        "Enterprise": {
            "Price": "Custom pricing",
            "Features": [
                "API access & integrations",
                "Analytics dashboard",
                "Dedicated support team"
            ]
        }
    }

    for name, info in plans.items():
        st.subheader(f"{name} — {info['Price']}")
        for feat in info["Features"]:
            st.write(f"• {feat}")
        st.button(f"Choose {name}")
        st.divider()

elif page == "📝 Feedback":
    st.title("📝 Feedback Form")
    st.write("We’d love to hear from you!")

    with st.form("feedback_form", clear_on_submit=True):
        name = st.text_input("Your Name")
        email = st.text_input("Your Email (optional)")
        feedback = st.text_area("Your Feedback")
        submitted = st.form_submit_button("Submit")
        if submitted:
            if name and feedback:
                st.success("✅ Thanks for your feedback!")
            else:
                st.error("Please provide at least your name and feedback.")
