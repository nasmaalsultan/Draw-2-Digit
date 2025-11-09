# Draw-2-Digit

Draw2Digit is an interactive Flask web app that recognizes hand-drawn digits (0–9) using a Convolutional Neural Network (CNN) trained on the MNIST dataset. Users can draw a digit in the browser, and the model predicts it along with confidence scores.

<img width="1200" height="400" alt="training_history" src="https://github.com/user-attachments/assets/0382a884-5fdb-47e1-b160-258f98ac38fa" />

---

## Features

- Draw any digit (0–9) directly in your browser  
- Digit recognition powered by TensorFlow/Keras  
- Displays confidence scores and prediction probabilities  
- Simple Flask backend and clean HTML/JS frontend  
- Exploratory NLP/ML project focused on model deployment and user interaction

---

## What I Learned

This project helped me:
- Understand how Convolutional Neural Networks process visual patterns.  
- Integrate a trained ML model with a Flask backend.  
- Build a frontend–backend pipeline that sends image data as Base64 and receives predictions via JSON.  
- Manage model training, data preprocessing, and deployment challenges (GPU/CPU compatibility, TensorFlow optimization).  

---

## Project Structure

Draw2Digit/
│
├── templates/
│ └── index.html # Frontend (drawing canvas + JS)
├── app.py # Flask app for serving and prediction
├── train_model.py # CNN training script (basic version)
├── train_model_fixed.py # Improved, GPU-safe training version
├── mnist_cnn_model.h5 # Trained model weights
├── training_history.png # Training/validation accuracy and loss
├── requirements.txt # Dependencies
├── README.md # Project documentation
└── LICENSE

---

## How It Works

- You draw a digit on the canvas.
- The canvas image is converted to Base64 and sent to Flask via /predict.
- The backend decodes, preprocesses, and normalizes the image (28×28 grayscale).
- The CNN model predicts the digit and returns a confidence score + all probabilities.
- The result is displayed dynamically in the browser.

---

## Tech Stack

- Python (Flask, TensorFlow, Keras, NumPy, Pillow)
- Frontend: HTML, CSS, JavaScript (Canvas API)
- Dataset: MNIST handwritten digits
- Visualization: Matplotlib

---

## Example

<img width="2880" height="1486" alt="image" src="https://github.com/user-attachments/assets/f6a8672d-c426-45ae-a429-9a9727112222" />

