<h2 style="font-family:Arial;">SVHN Digit Classifier (PyTorch + ONNX)</h2>


<h2 style="font-family:Verdana;">Overview</h2>


This project is a digit classification model trained on the SVHN (Street View House Numbers) dataset using PyTorch. The goal is to recognize digits (0–9) from real‑world images captured from house numbers. The project includes:

A trained PyTorch CNN model

An exported ONNX model

A simple web demo using JavaScript to run inference in the browser

A clean HTML interface for testing predictions

This project demonstrates skills in deep learning, model deployment, and web integration.

<h2 style="font-family:Verdana;">Live Demo</h2>



You can test the model directly in your browser:

👉 Live Website: https://svaughn26.github.io/svhn_torch/

<h2 style="font-family:Verdana;">Technologies Used</h2>



PyTorch — model training and architecture

ONNX — model export for cross‑platform inference

JavaScript (app.js) — client‑side inference logic

HTML/CSS — simple UI for testing predictions

SVHN Dataset — real‑world digit images

<h2 style="font-family:Verdana;">Model Architecture</h2>


The classifier uses a Convolutional Neural Network (CNN) with:

Two convolutional layers

ReLU activations

Max pooling

Fully connected layers

Output logits for 10 digit classes

The ONNX file (svhn_cnn.onnx) contains the exported version of the trained model.

<h2 style="font-family:Verdana;">Project Structure</h2>


**Code**
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


<h2 style="font-family:Verdana;">What I Learned</h2>



How to build and train CNNs using PyTorch

How to export models to ONNX for deployment

How to run machine learning inference in the browser

How to connect deep learning models with simple web interfaces

How to work with real‑world datasets like SVHN

<h2 style="font-family:Verdana;">Future Improvements</h2>



Add full training script

Add preprocessing examples

Improve UI styling

Add support for drawing digits directly on the page
