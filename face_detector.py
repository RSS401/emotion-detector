import cv2
import mediapipe as mp
import numpy as np


def detect_face(frame):
    mp_face_detection = mp.solutions.face_detection
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x = int(bboxC.xmin * iw)
                y = int(bboxC.ymin * ih)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)
                # Clamp coordinates to image bounds
                x = max(0, x)
                y = max(0, y)
                w = max(1, min(w, iw - x))
                h = max(1, min(h, ih - y))
                if w > 0 and h > 0 and y + h <= ih and x + w <= iw:
                    face_img = frame[y:y+h, x:x+w]
                    if face_img.size == 0:
                        continue
                    # Return RGB face image (no conversion to grayscale)
                    return face_img, (x, y, w, h)
        return None, None 