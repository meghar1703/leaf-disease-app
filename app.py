import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import io
import cv2

# Page Config
st.set_page_config(page_title="AI Plant Doctor 🌿", page_icon="🌿")


# Load Model

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()


# Load Class Names

with open("class_names.json") as f:
    class_names = json.load(f)


# Grad-CAM Function

def get_gradcam(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)

    if hasattr(heatmap, "numpy"):
        heatmap = heatmap.numpy()
    return heatmap


# Title

st.title("🌿 AI Plant Doctor")
st.write("Upload a leaf image to detect plant disease")


# Upload

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        file_bytes = uploaded_file.read()
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")

        st.image(image, caption="Uploaded Image")

        # Preprocess
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])

        # Output
        st.success(f"🌿 Prediction: {class_names[predicted_class]}")
        st.info(f"Confidence: {confidence:.2%}")

        # Confidence interpretation
        if confidence > 0.7:
            st.success("High confidence prediction ")
        elif confidence > 0.4:
            st.info("Moderate confidence")
        else:
            st.warning("Low confidence — try clearer image")

        # Top 3 predictions
        st.subheader("Top 3 Predictions")
        top_indices = np.argsort(predictions[0])[-3:][::-1]
        for i in top_indices:
            st.write(f"{class_names[i]} → {predictions[0][i]*100:.2f}%")

        
        # Grad-CAM
        
        st.subheader(" Model Attention (Grad-CAM)")

        
        last_conv_layer_name = "Conv_1"  # For MobileNetV2

        heatmap = get_gradcam(img_array, model, last_conv_layer_name)

        # Resize heatmap
        heatmap = cv2.resize(heatmap, (224, 224))
        heatmap = np.uint8(255 * heatmap)

        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Superimpose
        original = cv2.resize(np.array(image), (224, 224))
        superimposed = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

        st.image(superimposed, caption="Grad-CAM Heatmap")

    except Exception as e:
        st.error(f"Error: {e}")