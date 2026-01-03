import numpy as np
import os
import tensorflow as tf

# OPTIMIZATION: Force CPU usage and reduce memory overhead
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications import DenseNet201
from PIL import Image
from io import BytesIO
from app.config import MODEL_PATH, LABEL_MAP

class SkinCancerClassifier:
    def __init__(self):
        # FIX: Do NOT load the model here. 
        # Keep it None so the server starts instantly (consuming <100MB RAM).
        self.model = None

    def build_and_load_model(self):
        """
        Reconstructs the model architecture from code and loads weights.
        """
        print("Constructing model architecture (Lazy Load)...")
        try:
            # Recreate the exact architecture
            base_model = DenseNet201(
                include_top=False, 
                weights=None, 
                input_shape=(75, 100, 3)
            )
            
            self.model = Sequential([
                base_model,
                Flatten(),
                Dropout(0.5),
                Dense(512, activation='relu'),
                Dense(9, activation='softmax')
            ])
            
            # Compile
            self.model.compile(
                optimizer='sgd', 
                loss='categorical_crossentropy', 
                metrics=['accuracy']
            )

            # Load weights
            print(f"Loading weights from {MODEL_PATH}...")
            self.model.load_weights(MODEL_PATH)
            print("Model weights loaded successfully.")

        except Exception as e:
            print(f"Critical error loading model: {e}")
            raise RuntimeError("Could not load model weights.")

    def preprocess(self, image_data: bytes) -> np.ndarray:
        img = Image.open(BytesIO(image_data)).convert('RGB')
        img = img.resize((100, 75)) 
        img_array = np.array(img)
        img_array = (img_array - np.mean(img_array)) / np.std(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def predict(self, image_data: bytes):
        # FIX: Check if model is loaded ONLY when a request comes in
        if not self.model:
            self.build_and_load_model()
            
        processed_image = self.preprocess(image_data)
        
        predictions = self.model.predict(processed_image)
        predicted_class_idx = np.argmax(predictions, axis=1)[0]
        confidence = float(np.max(predictions))
        
        result = {
            "class_id": int(predicted_class_idx),
            "label": LABEL_MAP.get(predicted_class_idx, "Unknown"),
            "confidence": f"{confidence:.2%}",
            "raw_predictions": predictions.tolist()[0]
        }
        return result

# Singleton instance
classifier = SkinCancerClassifier()
