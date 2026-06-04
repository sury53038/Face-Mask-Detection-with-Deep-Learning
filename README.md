# 😷 Real-Time Face Mask Detection using MobileNetV2

A deep learning-based real-time face mask detection system built using **TensorFlow**, **MobileNetV2**, **OpenCV**, and **Python**. The model can detect whether a person is wearing a face mask through a webcam feed and display the prediction with confidence scores in real time.

---

## 📌 Features

- Real-time face detection using OpenCV Haar Cascade Classifier
- Face mask classification using Transfer Learning (MobileNetV2)
- Fine-tuned deep learning model for improved performance
- Live webcam prediction with confidence scores
- Lightweight and fast inference
- Achieved **98.87% test accuracy**

---

## 🛠️ Tech Stack

### Machine Learning
- Python
- TensorFlow / Keras
- MobileNetV2 (Transfer Learning)
- NumPy

### Computer Vision
- OpenCV

### Development Environment
- Google Colab
- VS Code

---

## 📂 Dataset

The model was trained on a face mask dataset with the following structure:

```text
face-mask-data/
│
└── data/
    ├── with_mask/
    └── without_mask/
```

Classes:

```text
1 → Mask
0 → No Mask
```

---

## 🧠 Model Architecture

### Base Model

- MobileNetV2 (ImageNet Weights)
- Include Top = False

### Classification Head

```text
GlobalAveragePooling2D
        ↓
Dense(128, ReLU)
        ↓
Dropout(0.3)
        ↓
Dense(1, Sigmoid)
```

### Input Shape

```text
128 × 128 × 3
```

### Image Normalization

```python
X = X / 255.0
```

---

## 📈 Training Results

### Initial Training

```text
Training Accuracy  : 99.30%
Validation Accuracy: 98.02%
```

### Fine-Tuning Results

```text
Test Accuracy: 98.87%
```

---

## 📊 Model Performance

### Confusion Matrix

```text
[[766   0]
 [ 17 728]]
```

*(Generated after correcting prediction thresholding and label mapping.)*

### Classification Capability

The model successfully distinguishes between:

✅ Face with Mask

✅ Face without Mask

### Known Limitations

The model may classify some face occlusions (such as hands, phones, or other objects covering the mouth and nose region) as masks. This occurs because the training dataset primarily contains masked and unmasked faces, causing the model to learn correlations related to facial occlusion.

---

## 🚀 Real-Time Detection Pipeline

```text
Webcam Feed
      ↓
Face Detection (OpenCV Haar Cascade)
      ↓
Face Cropping
      ↓
Resize (128 × 128)
      ↓
Normalization (/255)
      ↓
MobileNetV2 Model
      ↓
Mask / No Mask Prediction
      ↓
Display Result
```

---

## 📁 Project Structure

```text
FaceMaskDetection/
│
├── dataset/
│
├── model/
│   └── face_mask_detector_finetuned.h5
│
├── realtime_mask_detector.py
│
├── face_detection.py
│
├── requirements.txt
│
└── README.md
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/your-username/FaceMaskDetection.git

cd FaceMaskDetection
```

### Create Virtual Environment

```bash
python -m venv venv
```

Activate:

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / Mac

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Face Detection

```bash
python face_detection.py
```

---

## ▶️ Run Real-Time Mask Detection

```bash
python realtime_mask_detector.py
```

Press:

```text
Q
```

or

```text
ESC
```

to exit.

---

## 📦 Requirements

Main dependencies:

```text
tensorflow
opencv-python
numpy
h5py
keras
```

Install all using:

```bash
pip install -r requirements.txt
```

---

## 🔮 Future Improvements

- Replace Haar Cascade with MediaPipe Face Detection
- Deploy using Angular + Node.js + Express + FastAPI
- Convert model to TensorFlow.js for browser-side inference
- Improve robustness against occlusions (hands, phones, scarves, etc.)
- Deploy on cloud platforms such as Render, Railway, or AWS

---

## 👨‍💻 Author

**Surya Kant Singh**

Computer Science Engineering Student  
Aspiring Machine Learning & Data Science Engineer

---

## ⭐ If you found this project useful

Consider giving the repository a star ⭐