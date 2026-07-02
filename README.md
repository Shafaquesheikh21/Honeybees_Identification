# 🐝 Honeybee Species Identification using Computer Vision
![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-Web%20Application-black?logo=flask)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv)
## 📌 Overview
Honeybee species play a vital role in agriculture, biodiversity, and ecosystem sustainability. Manual identification of honeybee species requires expert knowledge and can be time-consuming. This project leverages Computer Vision and Deep Learning to automatically classify honeybee images into one of five different species using a pre-trained VGG16 Convolutional Neural Network (CNN).
The trained model is integrated into a Flask web application, allowing users to upload an image and instantly identify the honeybee species through a simple and interactive interface.
## 📑 Table of Contents
- [Overview](#-overview)
- [Features](#-features)
- [Honeybee Species](#-honeybee-species)
- [Technologies Used](#️-technologies-used)
- [Dataset](#-dataset)
- [Image Preprocessing](#-image-preprocessing)
- [Model Architecture](#-model-architecture)
- [Workflow](#️-workflow)
- [Web Application](#-web-application)
- [Results](#-results)
## 🚀 Features
- 🐝 Automatic honeybee species identification
- 🧠 Transfer Learning using VGG16
- 📷 Image preprocessing and augmentation
- 🌐 Flask-based web application
- 🎨 User-friendly interface
- ⚡ Real-time prediction
- 📈 Multi-class classification
## 📂 Honeybee Species
The model classifies honeybee images into the following five species:
| Species | Common Name |
|---------|-------------|
| Apis cerana indica | Indian Honey Bee |
| Apis dorsata | Giant Honey Bee |
| Apis florea | Dwarf Honey Bee |
| Apis mellifera | Western Honey Bee |
| Trigona | Stingless Bee |
<img width="641" height="558" alt="image" src="https://github.com/user-attachments/assets/42ebeda9-4e37-4764-ae4d-aab5e47b2b6b" />

## 🛠️ Technologies Used
- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Flask
- HTML
- CSS
- JavaScript

## 📊 Dataset
The dataset consists of labeled images belonging to five different honeybee species.
Dataset Classes
Apis cerana indica
Apis dorsata
Apis florea
Apis mellifera
Trigona
The dataset is divided into:
Training Set
Validation Set
Testing Set

## 🔄 Image Preprocessing
Before training the model, several preprocessing techniques were applied to improve model performance.
Image resizing
Image normalization
Image labeling
Data augmentation
Rotation
Horizontal flipping
Gaussian Blur (used for SVM-based experiments)
These preprocessing steps improve the robustness of the model by reducing overfitting and increasing dataset diversity.

## 🧠 Model Architecture
This project uses Transfer Learning with the VGG16 architecture.
The model pipeline includes:
Pre-trained VGG16 Feature Extractor
Convolution Layers
ReLU Activation
Dropout (0.2)
Fully Connected Layer
Softmax Classifier
<img width="492" height="246" alt="image" src="https://github.com/user-attachments/assets/8cc7e788-7c4e-423b-bab4-784e51a5af0b" />

## ⚙️ Workflow

Input Honeybee Image
        │
        ▼
Image Preprocessing
(Resize + Normalize)
        │
        ▼
Data Augmentation
        │
        ▼
Feature Extraction using VGG16
        │
        ▼
Fully Connected Layer
        │
        ▼
Softmax Classification
        │
        ▼
Predicted Honeybee Species

## 💻 Web Application
The trained deep learning model is deployed using Flask and integrated into a responsive website developed using HTML, CSS, and JavaScript.
Users can simply upload an image of a honeybee and receive the predicted species in real time.
## Website Preview
<img width="550" height="238" alt="image" src="https://github.com/user-attachments/assets/f1332eae-888f-47ca-a137-ce040d38248c" />

## 📦 Required Libraries
-TensorFlow
-Keras
-OpenCV
-NumPy
-Flask
-Matplotlib
-Scikit-learn
-Pandas
## 📈 Results
The proposed model successfully classifies honeybee images into five different species using transfer learning.
### Model Highlights
- ✅ Multi-class honeybee species classification
- ✅ Robust feature extraction using VGG16
- ✅ Improved generalization through data augmentation
- ✅ Real-time prediction using Flask
- ✅ Interactive web interface
## 🔮 Future Enhancements
- Deploy the application on the cloud (AWS, Render, or Hugging Face Spaces)
- Improve accuracy using EfficientNet or Vision Transformers
- Support additional honeybee species
- Enable real-time video-based honeybee detection
- Develop a mobile application
