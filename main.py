"""
Real-time Emotion Detection Application

This script runs the main emotion detection application using webcam input.
It loads a pre-trained emotion recognition model and displays real-time
emotion predictions on detected faces.
"""

import time

import cv2
import numpy as np

from emotion_model import load_emotion_model, predict_emotion
from face_detector import detect_face


def find_working_camera():
    """
    Find and return a working camera by testing different camera indices.
    
    Returns:
        cv2.VideoCapture: Working camera object or None if no camera found
    """
    for i in range(4):  # Test camera indices 0-3
        print(f"Trying camera index {i}...")
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"Camera {i} works!")
                return cap
            else:
                cap.release()
        else:
            print(f"Camera {i} not available")
    return None


def main():
    """Main function to run the emotion detection application."""
    print("Starting Emotion Detector...")
    
    # Load the pre-trained emotion recognition model
    print("Loading emotion model...")
    model = load_emotion_model()
    print("Model loaded successfully!")

    # Initialize camera
    print("Opening webcam...")
    cap = find_working_camera()
    if cap is None:
        print("Error: Could not find a working camera.")
        print("Please check camera permissions in System Preferences > Security & Privacy > Camera")
        return
    
    # Allow camera time to initialize
    print("Waiting for camera to initialize...")
    time.sleep(2)
    
    print("Webcam opened successfully!")
    print("Press 'q' to quit")

    # Main processing loop
    frame_count = 0
    while True:
        # Capture frame from camera
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Failed to capture frame {frame_count}")
            frame_count += 1
            if frame_count > 10:
                print("Too many failed frame captures. Exiting.")
                break
            time.sleep(0.1)
            continue

        frame_count = 0  # Reset counter on successful capture

        # Detect faces in the frame
        face_img, face_coords = detect_face(frame)
        if face_img is not None:
            try:
                # Predict emotion for detected face
                emotion = predict_emotion(model, face_img)
                x, y, w, h = face_coords
                
                # Draw bounding box around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Display predicted emotion above face
                cv2.putText(frame, emotion.capitalize(), (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            except Exception as e:
                print(f"Error predicting emotion: {e}")

        # Display the frame
        cv2.imshow('Emotion Detector', frame)
        
        # Check for quit command
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    print("Closing webcam...")
    cap.release()
    cv2.destroyAllWindows()
    print("Program finished.")


if __name__ == "__main__":
    main() 