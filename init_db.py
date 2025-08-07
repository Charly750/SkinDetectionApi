from pymongo import MongoClient

client = MongoClient("mongodb+srv://allsup1988:Sy78HCV93V4lETTX@cluster0.whxlukw.mongodb.net/")
db = client["skin_detection"]
lesions = db["lesions"]

# Données à insérer
data = [
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

# Insertion si collection vide
if lesions.count_documents({}) == 0:
    lesions.insert_many(data)
    print("Base initialisée avec succès.")
else:
    print("Les données existent déjà.")
