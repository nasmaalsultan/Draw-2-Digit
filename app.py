from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import io
import base64
import re

app = Flask(__name__)

# Load the trained model
model = keras.models.load_model('mnist_cnn_model.h5')

def preprocess_image(image_data):
    """
    Preprocess the drawn digit for prediction
    """
    # Remove data URL prefix
    image_data = re.sub('^data:image/.+;base64,', '', image_data)
    
    # Decode base64 image
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    
    # Convert to grayscale and resize to 28x28
    image = image.convert('L').resize((28, 28))
    
    # Convert to numpy array and invert colors (MNIST has white digits on black background)
    image_array = np.array(image)
    image_array = 255 - image_array  # Invert colors
    
    # Normalize and reshape for model
    image_array = image_array.astype('float32') / 255.0
    image_array = image_array.reshape(1, 28, 28, 1)
    
    return image_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image data from the request
        data = request.get_json()
        image_data = data['image']
        
        # Preprocess the image
        processed_image = preprocess_image(image_data)
        
        # Make prediction
        predictions = model.predict(processed_image)
        predicted_digit = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        # Get all probabilities
        probabilities = predictions[0].tolist()
        
        return jsonify({
            'success': True,
            'prediction': int(predicted_digit),
            'confidence': round(confidence * 100, 2),
            'probabilities': [round(p * 100, 2) for p in probabilities]
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/clear', methods=['POST'])
def clear_canvas():
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)  # Changed to port 5001