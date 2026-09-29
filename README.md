<h2 style="font-family:Arial;">SVHN Digit Classifier (PyTorch + ONNX)</h2>


**Overview**\n

This project is a digit classification model trained on the SVHN (Street View House Numbers) dataset using PyTorch. The goal is to recognize digits (0–9) from real‑world images captured from house numbers. The project includes:

A trained PyTorch CNN model

An exported ONNX model

A simple web demo using JavaScript to run inference in the browser

A clean HTML interface for testing predictions

This project demonstrates skills in deep learning, model deployment, and web integration.

**Live Demo**\n
You can test the model directly in your browser:

👉 Live Website: https://svaughn26.github.io/svhn_torch/

**Technologies Used**\n
PyTorch — model training and architecture

ONNX — model export for cross‑platform inference

JavaScript (app.js) — client‑side inference logic

HTML/CSS — simple UI for testing predictions

SVHN Dataset — real‑world digit images

**Model Architecture**\n
The classifier uses a Convolutional Neural Network (CNN) with:

Two convolutional layers

ReLU activations

Max pooling

Fully connected layers

Output logits for 10 digit classes

The ONNX file (svhn_cnn.onnx) contains the exported version of the trained model.

**Project Structure**\n
Code
svhn_torch/
│── app.js               # JavaScript inference logic
│── index.html           # Web demo UI
│── svhn_cnn.onnx        # Exported ONNX model
│── README.md            # Project documentation

*How to Run Locally*
1. Clone the repository
Code
git clone https://github.com/svaughn26/svhn_torch
cd svhn_torch

2. Open the demo
Simply open index.html in your browser.
No server required.

3. Test predictions
Upload digit images or use the provided UI to run inference through the ONNX model.

*What I Learned*\n
How to build and train CNNs using PyTorch

How to export models to ONNX for deployment

How to run machine learning inference in the browser

How to connect deep learning models with simple web interfaces

How to work with real‑world datasets like SVHN

**Future Improvements**\n
Add full training script

Add preprocessing examples

Improve UI styling

Add support for drawing digits directly on the page
