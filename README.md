# 🌿 Plant Disease Recognition System

An AI-powered system to identify diseases in plants from leaf images using Convolutional Neural Networks (CNN).

## 📜 Overview
The *Plant Disease Recognition System* is a deep learning-based application that helps identify diseases in plant leaves. The model is trained on a large dataset containing various plant diseases and healthy leaves, utilizing advanced convolutional neural networks to provide accurate predictions.

This application allows users to upload an image of a plant leaf, and it will analyze the image to detect any potential diseases. The system is implemented in Python using TensorFlow and deployed with Streamlit for an easy-to-use web interface.

## 🎯 Key Features
- Disease Detection: Upload a plant leaf image, and the model will predict the disease (or whether the plant is healthy).
- User-friendly Interface: A simple and intuitive interface created with Streamlit.
- Accurate Predictions: The model is trained on 38 different disease categories, providing high accuracy in disease identification.
- Real-time Inference: Results are displayed almost instantly, providing farmers and researchers with timely insights.

## 🗂️ Dataset
- Source: [Kaggle Plant Disease Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
- The dataset contains over 87,000 images categorized into 38 different classes, including both healthy and diseased plant leaves.<br>
         - Training Set: 70,295 images<br>
         - Validation Set: 17,572 images<br>
         - Test Set: 33 images

## 🚀 How It Works
1. Upload Image
- Navigate to the 'Disease Recognition' page in the sidebar.<br>
- Upload an image of a plant leaf to be analyzed.
2. Prediction
- Click the Predict button to run the model on the uploaded image.<br>
- The system will display the image along with the predicted disease (or healthy status) and relevant information.

## 📊 Model Architecture
The core model is built using Convolutional Neural Networks (CNN) and has been trained on a large dataset of plant leaves with 38 different disease categories. Here's a brief overview of the architecture:

- Input Size: 128x128 RGB images
- Convolutional Layers: 5 layers with increasing filter sizes
- MaxPooling Layers: After each convolutional block
- Dropout: Added for regularization to avoid overfitting
- Fully Connected Layers: To classify into 38 different classes
- Activation Function: ReLU for hidden layers, Softmax for the output layer
- Optimizer: Adam with a learning rate of 0.0001
- Loss Function: Categorical Crossentropy

## 📈 Model Performance
- Training Accuracy: 97.82%
- Validation Accuracy: 94.59%

## 👩‍💻 Tech Stack
- Framework: TensorFlow, Keras
- Frontend: Streamlit
- Language: Python
- Visualization: Matplotlib, Seaborn
- Data Processing: Numpy, OpenCV
