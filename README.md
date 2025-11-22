#  Aerial Object Classification & Detection

##  Project Overview
This Capstone Project delivers a Deep Learning solution for **Aerial Surveillance** and **Wildlife Monitoring**. It solves the critical challenge of distinguishing between biological targets (**Birds**) and technological threats (**Drones**) using aerial imagery.

The solution consists of:
1.  **Binary Classifier:** A Transfer Learning model (MobileNetV2) to classify images.
2.  **Object Detector:** A YOLOv8 model to locate objects with bounding boxes.
3.  **Web Application:** A user-friendly Streamlit interface for real-time analysis.

##  Project Structure
├── app.py                     # Main Streamlit Application
├── requirements.txt           # Python dependencies
├── transfer_legacy.h5         # Trained Classification Model (MobileNetV2)
├── best_yolo.pt               # Trained YOLOv8 Detection Model
└── README.md                  # Project Documentation

##  How to Run Locally

### 1. Prerequisites
Ensure you have Python installed. Clone this repository and install the required libraries:
pip install -r requirements.txt

### 2. Launch the App
Run the Streamlit application:
streamlit run app.py

##  Model Details

### Classification Model
* **Architecture:** MobileNetV2 (Transfer Learning)
* **Input Size:** 224x224
* **Accuracy:** >95% (on Validation Set)
* **Classes:** Bird vs. Drone

### Detection Model
* **Architecture:** YOLOv8 Nano
* **Training Epochs:** 25
* **Metric:** mAP50 (Mean Average Precision)

##  Results
The model successfully distinguishes between drones and birds in varying lighting conditions and angles, achieving high precision in avoiding false positives.

##  Tech Stack
* **Python**
* **TensorFlow / Keras**
* **Ultralytics YOLO**
* **Streamlit**
* **NumPy & Pandas**
