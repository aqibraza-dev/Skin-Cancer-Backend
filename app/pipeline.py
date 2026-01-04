import numpy as np
import os
import gc # Import Garbage Collector
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications import DenseNet201
from PIL import Image
from io import BytesIO
from app.config import MODEL_PATH, LABEL_MAP

# Suppress TF logs to keep Render logs clean
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

class SkinCancerClassifier:
    def __init__(self):
        self.model = None
        self.build_and_load_model()

    def build_and_load_model(self):
        print("Constructing DenseNet201 architecture...")
        try:
            # 1. Recreate architecture
            # Input shape: (75, 100, 3) -> Height=75, Width=100
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
            # Do not raise here, or the server will crash on startup.
            # Let it fail gracefully during prediction if needed.

    def preprocess(self, image_data: bytes) -> np.ndarray:
        """
        Preprocessing pipeline optimized for Render memory limits.
        """
        img = Image.open(BytesIO(image_data)).convert('RGB')
        
        # SAFETY NET: Even though React sends 100x75, we force it here 
        # to prevent crashes if someone bypasses the frontend.
        # PIL Resize is (Width, Height)
        if img.size != (100, 75):
            img = img.resize((100, 75)) 
        
        img_array = np.array(img)
        
        # Standardization (Z-score)
        # FIX: Handle division by zero if image is solid color
        std_dev = np.std(img_array)
        if std_dev > 0:
            img_array = (img_array - np.mean(img_array)) / std_dev
        else:
            # Fallback: just center the data
            img_array = img_array - np.mean(img_array)
        
        # Add batch dimension: (1, 75, 100, 3)
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array

    def predict(self, image_data: bytes):
        if not self.model:
            self.build_and_load_model()
            if not self.model:
                return {"error": "Model not loaded"}
            
        try:
            processed_image = self.preprocess(image_data)
            
            # Inference
            predictions = self.model.predict(processed_image)
            predicted_class_idx = np.argmax(predictions, axis=1)[0]
            confidence = float(np.max(predictions))
            
            result = {
                "class_id": int(predicted_class_idx),
                "label": LABEL_MAP.get(predicted_class_idx, "Unknown"),
                "confidence": f"{confidence:.2%}",
                # "raw_predictions": predictions.tolist()[0] # Comment out to save bandwidth
            }
            return result
            
        except Exception as e:
            return {"error": str(e)}
        
        finally:
            # CRITICAL FOR RENDER FREE TIER:
            # Force Python to release memory immediately after every request
            gc.collect()

# Singleton instance
classifier = SkinCancerClassifier()