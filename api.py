from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image, ImageEnhance
from PIL import ImageOps
import io, base64
import tflite_runtime.interpreter as tflite
import os
from pymongo import MongoClient
app = Flask(__name__)
CORS(app)
from pymongo import MongoClient, errors

MONGO_URI = "mongodb+srv://allsup1988:Sy78HCV93V4lETTX@cluster0.whxlukw.mongodb.net/skin_detection?retryWrites=true&w=majority"

try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)  # 5 secondes
    db = client["skin_detection"]
    lesions_collection = db["lesions"]
    # Test de connexion
    client.server_info()
    print("Connexion à MongoDB réussie")
except errors.ServerSelectionTimeoutError as e:
    print("Erreur de connexion MongoDB:", e)
    lesions_collection = None


# Charger modèle TFLite au démarrage
MODEL_PATH = os.path.join(os.path.dirname(__file__), "cbam_best_model_LeakyRelu_float32.tflite")
interpreter = tflite.Interpreter(model_path=MODEL_PATH) 

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


classes = [
    {
        "name": "actinic keratosis",
        "description": "Lésion précancéreuse causée par une exposition prolongée au soleil.",
        "image": "/images/actinic_keratosis/actinic_keratosis.jpg"
    },
    {
        "name": "basal cell carcinoma",
        "description": "Forme la plus courante de cancer de la peau, souvent localisée et peu agressive.",
        "image": "/images/basal_cell_carcinoma/basal_cell_carcinoma.jpg"
    },
    {
        "name": "dermatofibroma",
        "description": "Nodule bénin généralement brunâtre ou rougeâtre.",
        "image": "/images/dermatofibroma/dermatofibroma.jpg"
    },
    {
        "name": "melanoma",
        "description": "Cancer cutané très agressif, potentiellement mortel s’il n’est pas détecté tôt.",
        "image": "/images/melanoma/melanoma.jpg"
    },
    {
        "name": "nevus",
        "description": "Communément appelé grain de beauté, généralement bénin.",
        "image": "/images/nevus/nevus.jpg"
    },
    {
        "name": "pigmented benign keratosis",
        "description": "Lésion pigmentée bénigne, souvent confondue avec un mélanome.",
        "image": "/images/pigmented_benign_keratosis/pigmented_benign_keratosis.jpg"
    },
    {
        "name": "seborrheic keratosis",
        "description": "Croissance bénigne d’origine non cancéreuse de l'épiderme.",
        "image": "/images/seborrheic_keratosis/seborrheic_keratosis.jpg"
    },
    {
        "name": "squamous cell carcinoma",
        "description": "Deuxième type de cancer de la peau le plus fréquent, peut s’étendre si non traité.",
        "image": "/images/squamous_cell_carcinoma/squamous_cell_carcinoma.jpg"
    },
    {
        "name": "vascular lesion",
        "description": "Anomalie des vaisseaux sanguins, souvent bénigne.",
        "image": "/images/vascular_lesion/vascular_lesion.jpg"
    }
]

def preprocess_image(image, target_size=(128, 128)):
    # Corriger l'orientation EXIF (important pour iPhone)
    image = ImageOps.exif_transpose(image)
    
    # Redimensionner uniquement
    image = image.resize(target_size)
    
    # Convertir en array sans normalisation car modèle le fait déjà
    image_np = np.array(image).astype(np.float32)
    image_np = np.expand_dims(image_np, axis=0)
    return image_np


@app.route('/predict', methods=['POST'])
def make_prediction():
    if 'image_base64' not in request.json:
        return jsonify({"error": "Aucune image fournie"}), 400

    img_data = base64.b64decode(request.json['image_base64'])
    image = Image.open(io.BytesIO(img_data)).convert("RGB")
    image = image.resize((128, 128))
    input_array = preprocess_image(image)

    # Inference TFLite
    interpreter.set_tensor(input_details[0]['index'], input_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    class_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction))
    class_info = classes[class_index]

    return jsonify({
        "prediction": prediction.tolist(),
        "class_index": class_index,
        "class_name": class_info["name"],
        "confidence": confidence,
        "class_description": class_info["description"],
        "class_image": class_info["image"]
    })

@app.route('/classes', methods=['GET'])
def get_classes():
    if lesions_collection is None:
        return jsonify({"error": "Connexion à la base interrompue"}), 500

    try:
        lesions = list(lesions_collection.find({}, {"_id": 0}))
        return jsonify(lesions), 200
    except Exception as e:
        print("Erreur lors de la requête:", e)
        return jsonify({"error": "Impossible de récupérer les données"}), 500



@app.route('/')
def index():
    return "API TFLite opérationnelle !"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
