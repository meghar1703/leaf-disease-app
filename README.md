# 🌿 AI Plant Disease Detection System

## Live Demo
👉 https://huggingface.co/spaces/megha1703/leaf-disease-detection-app


## Overview

This project is a deep learning-based plant disease detection system that classifies leaf images into multiple disease categories.
The model is trained on a large dataset and deployed using Streamlit and Hugging Face Spaces.

##  Features

*  Multi-class plant disease classification
*  Confidence score display
* Top-3 predictions
* Grad-CAM visualization (model explainability)
* Live deployment

---

## Tech Stack

* Python
* TensorFlow / Keras
* Streamlit
* OpenCV
* Hugging Face Spaces
* Docker

## Model Details

* Transfer Learning (MobileNet / EfficientNet)
* Fine-tuned on plant disease dataset
* Achieved ~95% validation accuracy

## Real-World Insight
The model performs well on clean images but may show lower confidence on real-world inputs due to domain shift (lighting, background, noise).
This is handled using confidence thresholds and top-k predictions.

##  Future Improvements

* Improve real-world robustness
* Add more diverse dataset
* Optimize inference speed

## Author
Megha R
