# ======== ENVIRONMENT SETUP (MUST BE FIRST) ========
import os

# Disable GPU probing and suppress noisy logs
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ======== STANDARD IMPORTS ========
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications import DenseNet201
from PIL import Image
from io import BytesIO

from app.config import MODEL_PATH, LABEL_MAP


class SkinCancerClassifier:
    """
    Skin cancer image classifier using DenseNet201.
    Model is lazily initialized to avoid memory crashes on deployment platforms.
    """

    def __init__(self):
        self.model = None

    def _build_model(self) -> Sequential:
        """
        Reconstructs the exact training architecture.
        """
        base_model = DenseNet201(
            include_top=False,
            weights=None,
            input_shape=(75, 100, 3)
        )

        model = Sequential([
            base_model,
            Flatten(),
            Dropout(0.5),
            Dense(512, activation="relu"),
            Dense(9, activation="softmax")
        ])

        model.compile(
            optimizer="sgd",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        return model

    def _load_weights(self):
        """
        Loads trained weights into the model.
        """
        if not self.model:
            self.model = self._build_model()
            self.model.load_weights(MODEL_PATH)

    def preprocess(self, image_data: bytes) -> np.ndarray:
        """
        Image preprocessing identical to the notebook inference logic.
        """
        img = Image.open(BytesIO(image_data)).convert("RGB")
        img = img.resize((100, 75))

        img_array = np.array(img)
        img_array = (img_array - img_array.mean()) / img_array.std()

        return np.expand_dims(img_array, axis=0)

    def predict(self, image_data: bytes) -> dict:
        """
        Performs inference and returns prediction metadata.
        """
        self._load_weights()

        processed_image = self.preprocess(image_data)
        predictions = self.model.predict(processed_image)

        class_index = int(np.argmax(predictions, axis=1)[0])
        confidence = float(np.max(predictions))

        return {
            "class_id": class_index,
            "label": LABEL_MAP.get(class_index, "Unknown"),
            "confidence": f"{confidence:.2%}",
            "raw_predictions": predictions.tolist()[0]
        }


# ======== LAZY SINGLETON ACCESSOR ========

_classifier_instance: SkinCancerClassifier | None = None


def get_classifier() -> SkinCancerClassifier:
    """
    Provides a singleton classifier instance with lazy initialization.
    """
    global _classifier_instance

    if _classifier_instance is None:
        _classifier_instance = SkinCancerClassifier()

    return _classifier_instance
