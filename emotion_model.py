"""
Emotion Recognition Model Module

This module handles loading and using the pre-trained emotion recognition model.
It provides functions to load the model and predict emotions from face images.
"""

import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

# Emotion labels in the same order as the model's output classes
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Global variable to cache the loaded model
_model = None


def load_emotion_model():
    """
    Load and cache the emotion recognition model.
    
    Returns:
        tensorflow.keras.Model: Loaded emotion recognition model
    """
    global _model
    if _model is None:
        _model = load_model('emotion_model.h5')
    return _model


def predict_emotion(model, face_img):
    """
    Predict emotion from a face image.
    
    Args:
        model: Loaded emotion recognition model
        face_img: RGB face image (numpy array)
    
    Returns:
        str: Predicted emotion label
    """
    # Resize image to 224x224 for MobileNetV2 input
    img = cv2.resize(face_img, (224, 224))
    
    # Preprocess image for MobileNetV2
    img = preprocess_input(img)
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)  # Shape: (1, 224, 224, 3)
    
    # Get model predictions
    preds = model.predict(img, verbose=0)
    
    # Return the emotion with highest probability
    emotion_idx = np.argmax(preds)
    return EMOTION_LABELS[emotion_idx] 