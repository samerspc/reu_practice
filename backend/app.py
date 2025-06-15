from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)
CORS(app)

model = tf.keras.models.load_model('../model_mobilenet_final.h5')

FOOD_CATEGORIES = [
    "apple", "banana", "carrot", "orange", "pizza", "sandwich", "strawberry"
]

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        image_file = request.files['image']
        image_bytes = image_file.read()
        
        processed_image = preprocess_image(image_bytes)
        
        predictions = model.predict(processed_image)
        
        top_prediction_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][top_prediction_idx])
        predicted_class = FOOD_CATEGORIES[top_prediction_idx]
        
        return jsonify({
            'prediction': predicted_class,
            'confidence': confidence
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 