import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tflite_runtime.interpreter as tflite
import os

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="EcoSortAI TFLite", page_icon="♻️")

# =========================
# MODEL
# =========================
MODEL_PATH = "recycling_model.tflite"
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

# Load TFLite model
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# =========================
# HELPERS
# =========================
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = ImageOps.fit(image, IMG_SIZE, Image.Resampling.LANCZOS)
    arr = np.asarray(image).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_classes(image: Image.Image):
    x = preprocess_image(image)
    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    top_idx = int(np.argmax(output_data))
    return ALL_CLASSES[top_idx], float(output_data[top_idx])

# =========================
# STREAMLIT UI
# =========================
st.title("♻️ EcoSortAI TFLite")
st.write("Upload an image and EcoSortAI will classify it into one of 27 categories.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    predicted_class, prob = predict_classes(image)
    st.subheader(f"Prediction: **{predicted_class}** ({prob*100:.2f}%)")
