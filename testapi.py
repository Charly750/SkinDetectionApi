import os
import base64
import requests

# Chemin vers le dossier contenant les images classées par classe
root_dir = "C:/Users/ultim/Desktop/MEMOIRE/cas_concret/app/back/images"
api_url = "http://127.0.0.1:5000/predict"

# Parcours de chaque classe (chaque sous-dossier)
for class_folder in os.listdir(root_dir):
    class_path = os.path.join(root_dir, class_folder)

    if os.path.isdir(class_path):
        # Trouver une image dans le dossier
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not image_files:
            print(f"❌ Aucune image trouvée pour {class_folder}")
            continue

        image_path = os.path.join(class_path, image_files[0])  # Prend la première image
        print(f"\n📷 Test image: {image_path} (Attendu: {class_folder})")

        # Lecture de l'image et encodage base64
        with open(image_path, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode('utf-8')

        # Envoi à l'API
        response = requests.post(api_url, json={"image_base64": img_base64})
        if response.status_code == 200:
            result = response.json()
            predicted_class = result.get("class_name", "??")
            confidence = result.get("confidence", 0)
            print(f"✅ Prédit: {predicted_class} (confiance: {confidence:.2%})")
        else:
            print(f"❌ Erreur API: {response.status_code} - {response.text}")
