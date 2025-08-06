from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
from PIL import Image
import io
import base64
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras import layers

# ---- CBAM (inchangé) ----
class CBAM(tf.keras.layers.Layer):
    def __init__(self, channels, reduction_ratio=8, **kwargs):
        super(CBAM, self).__init__(**kwargs)
        self.channels = channels
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        self.global_avg_pool = layers.GlobalAveragePooling2D()
        self.fc1 = layers.Dense(self.channels // self.reduction_ratio, activation='relu')
        self.fc2 = layers.Dense(self.channels, activation='sigmoid')
        self.spatial_conv = layers.Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')

    def call(self, inputs):
        avg_pool = self.global_avg_pool(inputs)
        dense = self.fc1(avg_pool)
        dense = self.fc2(dense)
        channel_att = tf.reshape(dense, [-1, 1, 1, self.channels])
        x = inputs * channel_att

        avg = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_ = tf.reduce_max(x, axis=-1, keepdims=True)
        concat = tf.concat([avg, max_], axis=-1)
        spatial_att = self.spatial_conv(concat)
        return x * spatial_att

    def get_config(self):
        config = super().get_config()
        config.update({
            "channels": self.channels,
            "reduction_ratio": self.reduction_ratio
        })
        return config

# ---- API ----
app = Flask(__name__)
CORS(app)

# Chargement du modèle
model_path = "cbam_best_model.keras"
model = load_model(model_path, custom_objects={'CBAM': CBAM})

# Liste des noms de classes (à adapter selon ton problème)
class_names = ['actinic keratosis', 'basal cell carcinoma', 'dermatofibroma', 'melanoma', 'nevus', 'pigmented benign keratosis', 'seborrheic keratosis', 'squamous cell carcinoma', 'vascular lesion']

def preprocess_image(image, target_size=(128, 128)):
    """Redimensionne et normalise l'image pour le modèle"""
    image = image.resize(target_size)
    image = np.array(image).astype("float32")
    if len(image.shape) == 2:  # si image en niveaux de gris
        image = np.expand_dims(image, axis=-1)
        image = np.repeat(image, 3, axis=-1)
    return np.expand_dims(image, axis=0)  # batch size 1

@app.route('/predict', methods=['POST'])
def make_prediction():
    # Vérifier si l'image est envoyée en base64
    if 'image_base64' not in request.json:
        return jsonify({"error": "Aucune image fournie"}), 400

    img_data = base64.b64decode(request.json['image_base64'])
    image = Image.open(io.BytesIO(img_data)).convert("RGB")

    # Prétraitement et prédiction
    input_array = preprocess_image(image)
    prediction = model.predict(input_array)

    # Trouver la classe prédite
    class_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction))
    class_name = class_names[class_index]

    return jsonify({
        "prediction": prediction.tolist(),
        "class_index": class_index,
        "class_name": class_name,
        "confidence": confidence
    })

@app.route('/')
def index():
    return "Bienvenue sur l'API de prédiction (images) ! Envoyez une requête POST à /predict avec une image en base64."

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
