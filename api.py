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

client = MongoClient("mongodb+srv://allsup1988:Sy78HCV93V4lETTX@cluster0.whxlukw.mongodb.net/")
db = client["skin_detection"]
lesions = db["lesions"]

# Charger modèle TFLite au démarrage
MODEL_PATH = os.path.join(os.path.dirname(__file__), "cbam_best_model_LeakyRelu_float32.tflite")
interpreter = tflite.Interpreter(model_path=MODEL_PATH) 

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = [
    'actinic keratosis', 'basal cell carcinoma', 'dermatofibroma',
    'melanoma', 'nevus', 'pigmented benign keratosis',
    'seborrheic keratosis', 'squamous cell carcinoma', 'vascular lesion'
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
    return jsonify({
        "prediction": prediction.tolist(),
        "class_index": class_index,
        "class_name": class_names[class_index],
        "confidence": confidence
    })

@app.route('/classes', methods=['GET'])
def get_classes():
    try:
        lesions = list(lesions_collection.find({}, {"_id": 0}))
        if not lesions:
            return jsonify({"message": "Aucune donnée disponible dans la base."}), 204
        return jsonify(lesions), 200
    except Exception as e:
        print(f"Erreur MongoDB : {e}")
        return jsonify({"error": "Erreur lors de la récupération des classes."}), 500


@app.route('/')
def index():
    return "API TFLite opérationnelle !"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
