import os
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from flask_cors import CORS

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# Define model paths and their properties
MODELS = {
    "mobilenet_final": {
        "path": 'model_mobilenet_final.h5',
        "img_size": 128,
        "labels": [
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
        ],
        "model": None # Will be loaded at app startup
    },
    "food101_discriminator": {
        "path": 'GAN_model/food101_discriminator.h5',
        "img_size": 64,
        "labels": [
            "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare",
            "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito",
            "broccoli_cheese_soup", "bruschetta", "buffalo_chicken_nachos", "buffalo_wings",
            "caesar_salad", "cannoli", "caprese_salad", "carrot_cake", "ceviche",
            "cheesecake", "cheese_plate", "chicken_curry", "chicken_quesadilla",
            "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder",
            "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame", "cup_cakes",
            "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict", "escargots",
            "falafel", "filet_mignon", "fish_and_chips", "foie_gras", "french_fries",
            "french_onion_soup", "french_toast", "fried_calamari", "fried_rice", "frozen_yogurt",
            "garlic_bread", "gnocchi", "greek_salad", "grilled_cheese_sandwich",
            "grilled_salmon", "guacamole", "gyoza", "hamburger", "hot_dog", "huevos_rancheros",
            "hummus", "ice_cream", "lasagna", "lobster_bisque", "lobster_roll_sandwich",
            "macaroni_and_cheese", "macarons", "miso_soup", "mussels", "nachos", "omelette",
            "onion_rings", "oysters", "pad_thai", "paella", "pancakes", "panna_cotta",
            "peking_duck", "pho", "pizza", "pork_chop", "poutine", "prime_rib", "pulled_pork_sandwich",
            "ramen", "ravioli", "red_velvet_cake", "risotto", "samosa", "sashimi", "scallops",
            "shrimp_scampi", "smoked_salmon", "sushi", "tacos", "takoyaki", "tiramisu",
            "tuna_tartare", "waffles"
        ],
        "model": None # Will be loaded at app startup
    }
}

# Load models at startup
for model_name, props in MODELS.items():
    try:
        props["model"] = load_model(props["path"])
        print(f"Successfully loaded model: {model_name} from {props['path']}")
    except Exception as e:
        print(f"Error loading model {model_name} from {props['path']}: {e}")
        # Handle case where model might be missing, e.g., set a flag or remove from MODELS


@app.route('/', methods=['GET'])
def home():
    return "Welcome to the Food Recognition API! Use the /predict endpoint to upload images."

@app.route('/predict', methods=['POST'])
def predict():
    model_name = request.form.get('model_name', 'mobilenet_final') # Default to mobilenet_final

    if model_name not in MODELS or MODELS[model_name]["model"] is None:
        return jsonify({'error': f'Model \'{model_name}\' not found or not loaded.'}), 400

    selected_model_props = MODELS[model_name]
    model = selected_model_props["model"]
    img_size = selected_model_props["img_size"]
    class_labels = selected_model_props["labels"]

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        try:
            img = Image.open(file.stream).convert('RGB')
            img = img.resize((img_size, img_size))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) # Create a batch

            # Apply specific preprocessing based on the model
            if model_name == "mobilenet_final":
                img_array /= 255.0 # Rescale to [0, 1]
            elif model_name == "food101_discriminator":
                img_array = (img_array / 127.5) - 1.0 # Normalize to [-1, 1]

            predictions = model.predict(img_array)

            # For the discriminator, predictions is a list: [validity, labels]
            if model_name == "food101_discriminator":
                predicted_class_index = np.argmax(predictions[1][0]) # Take labels prediction
                # Use the validity prediction for confidence, or just the label confidence
                confidence = float(predictions[1][0][predicted_class_index])
            else:
                predicted_class_index = np.argmax(predictions[0])
                confidence = float(np.max(predictions[0]))

            predicted_food = class_labels[predicted_class_index]

            return jsonify({
                'predicted_food': predicted_food,
                'confidence': confidence,
                'model_used': model_name
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0') 