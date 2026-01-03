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
        Reconstructs the model architecture from code and loads weights.
        This bypasses config loading errors (AttributeError, batch_shape, etc).
        """
        print("Constructing model architecture...")
        try:
            # 1. Recreate the exact architecture from your notebook
            # Input shape: (75, 100, 3) -> (Height, Width, Channels)
            base_model = DenseNet201(
                include_top=False, 
                weights=None,  # We will load our own weights
                input_shape=(75, 100, 3)
            )
            
            self.model = Sequential([
                base_model,
                Flatten(),
                Dropout(0.5),
                Dense(512, activation='relu'),
                Dense(9, activation='softmax') # 9 Classes
            ])
            
            # 2. Compile (needed to initialize variables properly)
            self.model.compile(
                optimizer='sgd', 
                loss='categorical_crossentropy', 
                metrics=['accuracy']
            )

            # 3. Load the weights from the h5 file
            print(f"Loading weights from {MODEL_PATH}...")
            self.model.load_weights(MODEL_PATH)
            print("Model weights loaded successfully.")

        except Exception as e:
            print(f"Critical error loading model: {e}")
            raise RuntimeError("Could not load model weights.")

    def preprocess(self, image_data: bytes) -> np.ndarray:
        """
        Matches the preprocessing from your notebook's prediction cell.
        """
        img = Image.open(BytesIO(image_data)).convert('RGB')
        
        # Notebook: resize((100, 75)) -> (Width, Height)
        img = img.resize((100, 75)) 
        
        img_array = np.array(img)
        
        # Notebook Cell 34 Inference Logic: Per-image standardization
        img_array = (img_array - np.mean(img_array)) / np.std(img_array)
        
        # Add batch dimension: (1, 75, 100, 3)
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array

    def predict(self, image_data: bytes):
        if not self.model:
            self.build_and_load_model()
            
        processed_image = self.preprocess(image_data)
        
        # Inference
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