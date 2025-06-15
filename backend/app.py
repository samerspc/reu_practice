import os
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from flask_cors import CORS

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# Load your trained model
MODEL_PATH = 'model_mobilenet_final.h5'
model = load_model(MODEL_PATH)

# IMPORTANT: Replace this with your actual class labels in the correct order
# You can get this from train_gen.class_indices in your notebook
class_labels = [
    "apple pie", "baby back ribs", "baklava", "beef carpaccio", "beef tartare",
    "beet salad", "beignets", "bibimbap", "bread pudding", "breakfast burrito",
    "broccoli cheese soup", "bruschetta", "buffalo chicken nachos", "buffalo wings",
    "caesar salad", "cannoli", "caprese salad", "carrot cake", "ceviche",
    "cheesecake", "cheese plate", "chicken curry", "chicken quesadilla",
    "chicken wings", "chocolate cake", "chocolate mousse", "churros", "clam chowder",
    "club sandwich", "crab cakes", "creme brulee", "croque madame", "cup cakes",
    "deviled eggs", "donuts", "dumplings", "edamame", "eggs benedict", "escargots",
    "falafel", "filet mignon", "fish and chips", "foie gras", "french fries",
    "french onion soup", "french toast", "fried calamari", "fried rice", "frozen yogurt",
    "garlic bread", "gnocchi", "greek salad", "grilled cheese sandwich",
    "grilled salmon", "guacamole", "gyoza", "hamburger", "hot dog", "huevos rancheros",
    "hummus", "ice cream", "lasagna", "lobster bisque", "lobster roll sandwich",
    "macaroni and cheese", "macarons", "miso soup", "mussels", "nachos", "omelette",
    "onion rings", "oysters", "pad thai", "paella", "pancakes", "panna cotta",
    "peking duck", "pho", "pizza", "pork chop", "poutine", "prime rib", "pulled pork sandwich",
    "ramen", "ravioli", "red velvet cake", "risotto", "samosa", "sashimi", "scallops",
    "shrimp scampi", "smoked salmon", "sushi", "tacos", "takoyaki", "tiramisu",
    "tuna tartare", "waffles"
]


IMG_SIZE = 128

@app.route('/', methods=['GET'])
def home():
    return "Welcome to the Food Recognition API! Use the /predict endpoint to upload images."

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        try:
            img = Image.open(file.stream).convert('RGB')
            img = img.resize((IMG_SIZE, IMG_SIZE))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) # Create a batch
            img_array /= 255.0 # Rescale

            predictions = model.predict(img_array)
            predicted_class_index = np.argmax(predictions[0])
            predicted_food = class_labels[predicted_class_index]
            confidence = float(np.max(predictions[0]))

            return jsonify({
                'predicted_food': predicted_food,
                'confidence': confidence
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0') 