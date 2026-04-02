# 🌿 AI Plant Disease Detection System

## Live Demo
https://huggingface.co/spaces/YOUR-USERNAME/YOUR-APP

## Overview

This project presents an end-to-end deep learning system for **plant disease detection from leaf images**.
It leverages transfer learning to classify images into multiple disease categories and is deployed as a fully interactive web application.
This project focuses on **real-world usability**, addressing challenges such as domain shift, uncertainty in predictions, and model interpretability.

## Objectives

* Build a robust image classification model for plant disease detection
* Deploy the model as a user-friendly web application
* Handle real-world variability in inputs
* Provide interpretability using Grad-CAM
* Improve user trust with confidence-based outputs

## Key Features

###  Multi-Class Classification
* Classifies leaf images into **38 categories** including healthy and diseased classes
  
### Confidence-Aware Predictions
* Displays prediction probability
* Provides **confidence interpretation** (high / moderate / low)

### Top-3 Predictions
* Shows top-3 most likely classes
* Helps users understand model uncertainty

### Grad-CAM

* Visualizes **where the model is focusing** in the image
* Improves transparency and trust

### Deployment
* Fully deployed using Streamlit + Hugging Face Spaces
* Accessible via browser (no setup required)

## Model Architecture
* Base Model: Pre-trained CNN (e.g., MobileNetV2)
* Transfer Learning applied with frozen base layers
* Custom classification head:
  * Global Average Pooling
  * Dense layers with Dropout
* Fine-tuning applied to deeper layers for better generalization

## Training Strategy

### Data Processing
* Image resizing (224×224)
* Normalization / preprocessing aligned with base model

###  Data Augmentation
To simulate real-world conditions:
* Rotation
* Zoom
* Brightness variation
* Horizontal flipping
* Shift transformations

### Training Phases
1. Feature extraction (frozen base model)
2. Fine-tuning (unfreezing deeper layers)

### Loss Function
* Categorical Crossentropy (with label smoothing)

## Performance
* Validation Accuracy: **~95%**
* Strong performance on structured dataset images
* Variable confidence on real-world inputs due to domain shift

## Real-World Challenges & Solutions

### Challenge: Domain Shift
Real-world images differ from training data in:
* Lighting conditions
* Background noise
* Image quality

### Solutions Implemented
* Confidence-based feedback system
* Top-k predictions instead of single output
* Grad-CAM for visual explanation

This ensures the system remains **informative even under uncertainty**.

##  Sample Outputs
![PHOTO-2026-04-03-00-09-55](https://github.com/user-attachments/assets/b5253b57-065e-497f-8d8f-dc27e9040313)
### Prediction Output
![PHOTO-2026-04-03-00-12-02](https://github.com/user-attachments/assets/677d30c3-5f5c-46b0-9cc0-cc69d2420653)
### Grad-CAM Visualization
![PHOTO-2026-04-03-00-13-20](https://github.com/user-attachments/assets/24fba0cb-bf8d-41f0-9364-1aca2ea174fa)

## Tech Stack

* **Programming:** Python
* **Deep Learning:** TensorFlow / Keras
* **Frontend:** Streamlit
* **Image Processing:** OpenCV, PIL
* **Deployment:** Hugging Face Spaces
* **Containerization:** Docker

## Project Structure

```text
leaf-disease-detection/
│
├── app.py                # Streamlit application
├── model.h5             # Trained model
├── class_names.json     # Class labels
├── Dockerfile           # Deployment config
├── .streamlit/          # Streamlit config
├── README.md
└── screenshots/
```
## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
## Future Improvements
* Improve robustness with more real-world data
* Optimize model for faster inference
* Add mobile camera integration
* Deploy lightweight version for edge devices

## Key Takeaway
This project goes beyond traditional model training by focusing on:
* Real-world deployment
* Handling uncertainty
* Explainable AI (Grad-CAM)
* End-to-end ML system design





