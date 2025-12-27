# 😷 Face Mask Detection with Deep Learning

A lightweight, high-accuracy Deep Learning model capable of detecting whether a person is wearing a face mask or not. Built from scratch using **TensorFlow/Keras** and trained on a balanced dataset of masked and unmasked faces.

## 📖 Overview
The COVID-19 pandemic highlighted the importance of face masks in public safety. This project implements a **Convolutional Neural Network (CNN)** to classify images into two categories:
* **Mask** (The person is wearing a mask)
* **No Mask** (The person is not wearing a mask)

The model handles data preprocessing, normalization, and achieves a validation accuracy of **~92%**.

## ✨ Key Features
* **Custom CNN Architecture:** A sequential model with 3 convolutional blocks designed for feature extraction.
* **Optimized Data Pipeline:** Uses `tf.data.Dataset` for efficient buffering, caching, and prefetching.
* **Binary Classification:** Uses a Sigmoid activation function for precise probability scoring.
* **Overfitting Control:** Implements Dropout layers and Early Stopping to ensure generalization.
* **Visual Prediction:** Custom prediction script that visualizes the input image with color-coded confidence scores (Green for Mask, Red for No Mask).

## 🛠️ Tech Stack
* **Deep Learning:** TensorFlow, Keras
* **Data Processing:** NumPy, Pandas
* **Image Processing:** OpenCV, PIL (Pillow)
* **Visualization:** Matplotlib

## 📂 Dataset
The dataset consists of **7,553 images** divided into two classes.
* **With Mask:** ~3,725 images
* **Without Mask:** ~3,828 images

*Note: The dataset is perfectly balanced to prevent model bias.*

## 🧠 Model Architecture
The model consists of the following layers:
1.  **Input Layer:** 150x150x3 (RGB Images)
2.  **Conv2D + MaxPooling:** 32 filters (Feature Extraction)
3.  **Conv2D + MaxPooling:** 64 filters
4.  **Conv2D + MaxPooling:** 128 filters
5.  **Flatten Layer:** Converts 2D maps to 1D vector
6.  **Dense Layer:** 128 neurons (ReLU activation)
7.  **Dropout (0.5):** Randomly drops 50% of neurons to prevent overfitting
8.  **Output Layer:** 1 neuron (Sigmoid activation for binary output)

## 📊 Performance
* **Training Accuracy:** ~95%
* **Validation Accuracy:** ~92%
* **Loss Function:** Binary Crossentropy
* **Optimizer:** Adam

## 🚀 Installation & Usage

### 1. Clone the Repository
```bash
git clone [https://github.com/sury53038/Face-Mask-Detection-with-Deep-Learning.git](https://github.com/sury53038/Face-Mask-Detection-with-Deep-Learning.git)
cd Face-Mask-Detection-with-Deep-Learning
