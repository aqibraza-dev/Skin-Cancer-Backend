import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications import DenseNet201
from PIL import Image
from io import BytesIO
from app.config import MODEL_PATH, LABEL_MAP

# Suppress TF logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

class SkinCancerClassifier:
    def __init__(self):
        self.model = None
        self.build_and_load_model()

    def build_and_load_model(self):
        """
        Reconstructs the model architecture and loads weights.
        """
        print("Constructing model architecture...")
        try:
            # 1. Recreate the architecture
            # Input shape (75, 100, 3) usually implies Height=75, Width=100
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
            
            # 2. Compile
            self.model.compile(
                optimizer='sgd', 
                loss='categorical_crossentropy', 
                metrics=['accuracy']
            )

            # 3. Load weights
            print(f"Loading weights from {MODEL_PATH}...")
            self.model.load_weights(MODEL_PATH)
            print("Model weights loaded successfully.")

        except Exception as e:
            print(f"Critical error loading model: {e}")
            # We don't raise here so the API can still start (health checks pass)
            # but predict() will fail if model is None.

    def preprocess(self, image_data: bytes) -> np.ndarray:
        img = Image.open(BytesIO(image_data)).convert('RGB')
        
        # PIL resize is (Width, Height). 
        # If target is (75, 100, 3), we need Height=75, Width=100.
        img = img.resize((100, 75)) 
        
        img_array = np.array(img)
        
        # Standardization (Z-score normalization)
        std_dev = np.std(img_array)
        
        # Fix: Prevent division by zero if image is solid color
        if std_dev > 0:
            img_array = (img_array - np.mean(img_array)) / std_dev
        else:
            img_array = img_array - np.mean(img_array)
        
        # Add batch dimension: (1, 75, 100, 3)
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array

    def predict(self, image_data: bytes):
        if not self.model:
            self.build_and_load_model()
            if not self.model:
                raise RuntimeError("Model failed to initialize.")
            
        processed_image = self.preprocess(image_data)
        
        predictions = self.model.predict(processed_image)
        predicted_class_idx = np.argmax(predictions, axis=1)[0]
        confidence = float(np.max(predictions))
        
        result = {
            "class_id": int(predicted_class_idx),
            "label": LABEL_MAP.get(predicted_class_idx, "Unknown"),
            "confidence": f"{confidence:.2%}",
            # "raw_predictions": predictions.tolist()[0] 
        }
        return result

# Singleton instance
classifier = SkinCancerClassifier()
