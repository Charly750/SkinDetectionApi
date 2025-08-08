from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image, ImageEnhance
import io, base64
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# Charger modèle TFLite au démarrage
import os
MODEL_PATH = os.path.join(os.path.dirname(__file__), "cbam_best_model_LeakyRelu_float32.tflite")
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = [
    'actinic keratosis', 'basal cell carcinoma', 'dermatofibroma',
    'melanoma', 'nevus', 'pigmented benign keratosis',
    'seborrheic keratosis', 'squamous cell carcinoma', 'vascular lesion'
]

def preprocess_image(image, target_size=(128, 128)):
    # Ajustement luminosité et contraste
    image = ImageEnhance.Brightness(image).enhance(1.2)
    image = ImageEnhance.Contrast(image).enhance(1.3)
    image = image.resize(target_size)
    image_np = np.array(image).astype(np.float32)
    image_np = np.expand_dims(image_np, axis=0)
    return image_np

@app.route('/predict', methods=['POST'])
def make_prediction():
    if 'image_base64' not in request.json:
        return jsonify({"error": "Aucune image fournie"}), 400

    img_data = base64.b64decode(request.json['image_base64'])
    image = Image.open(io.BytesIO(img_data)).convert("RGB")
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

@app.route('/')
def index():
    return "API TFLite opérationnelle !"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
