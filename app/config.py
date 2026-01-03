import os
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "models/skin_disease_model.h5")

# Class mapping from your notebook
LABEL_MAP = {
    0: 'Pigmented Benign Keratosis',
    1: 'Melanoma',
    2: 'Vascular Lesion',
    3: 'Actinic Keratosis',
    4: 'Squamous Cell Carcinoma',
    5: 'Basal Cell Carcinoma',
    6: 'Seborrheic Keratosis',
    7: 'Dermatofibroma',
    8: 'Nevus'
}