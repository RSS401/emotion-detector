import numpy as np


def preprocess_face(face_img):
    # Example: normalize and expand dims for model
    face_img = face_img / 255.0
    face_img = np.expand_dims(face_img, axis=0)
    return face_img 