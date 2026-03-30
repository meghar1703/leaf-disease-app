import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import json
import cv2

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="AI Plant Doctor 🌿", layout="centered")

st.title("🌿 AI Plant Doctor")
st.write("Upload a leaf image to detect plant disease")

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()

# -------------------------------
# LOAD CLASS NAMES
# -------------------------------
with open("class_names.json") as f:
    class_names = json.load(f)

# -------------------------------
# PREPROCESS IMAGE
# -------------------------------
def preprocess(image):
    image = image.resize((224, 224))
    img = np.array(image) / 255.0
    return np.expand_dims(img, axis=0)

# -------------------------------
# GRAD-CAM FUNCTIONS
# -------------------------------
def get_gradcam_heatmap(model, image, layer_name="Conv_1"):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_output, preds = grad_model(image)
        class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / (np.max(heatmap) + 1e-8)

    return heatmap

def overlay_heatmap(heatmap, image):
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

# -------------------------------
# FILE UPLOAD
# -------------------------------
uploaded_file = st.file_uploader("📤 Upload Leaf Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = preprocess(image)

    # Prediction
    preds = model.predict(img)
    idx = int(np.argmax(preds))
    confidence = float(np.max(preds))

    st.success(f"Prediction: {class_names[idx]}")
    st.write(f"Confidence: {confidence:.2%}")

    # -------------------------------
    # TOP 3 PREDICTIONS
    # -------------------------------
    st.subheader("🔍 Top 3 Predictions")
    top3 = preds[0].argsort()[-3:][::-1]

    for i in top3:
        st.write(f"{class_names[i]} → {preds[0][i]:.2%}")
        st.progress(float(preds[0][i]))

    # -------------------------------
    # GRAD-CAM
    # -------------------------------
    st.subheader("🔥 Model Focus (Grad-CAM)")

    img_cv = np.array(image)
    heatmap = get_gradcam_heatmap(model, img)
    gradcam_img = overlay_heatmap(heatmap, img_cv)

    st.image(gradcam_img, caption="Highlighted disease regions")
