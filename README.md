# Emotion Detector using Webcam

A real-time emotion detection system that uses your webcam to recognize facial expressions and classify them into 7 different emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.

## Features

- **Real-time emotion detection** from webcam feed
- **MobileNetV2-based deep learning model** for accurate emotion recognition
- **MediaPipe face detection** for robust face localization
- **Modular architecture** with separate components for face detection, emotion prediction, and training
- **Cross-platform compatibility** (Windows, macOS, Linux)
- **Easy setup** with virtual environment and requirements file

## Project Structure

```
emotion_detector/
├── main.py                    # Main application script
├── face_detector.py           # Face detection using MediaPipe
├── emotion_model.py           # Emotion recognition model loader and predictor
├── train_emotion_model.py     # Script to train custom emotion recognition model
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore rules
├── README.md                  # This file
├── emotion_model.h5           # Trained model (generated after training)
└── data/                      # Training and test datasets
    ├── train/
    │   ├── angry/
    │   ├── disgust/
    │   ├── fear/
    │   ├── happy/
    │   ├── neutral/
    │   ├── sad/
    │   └── surprise/
    └── test/
        ├── angry/
        ├── disgust/
        ├── fear/
        ├── happy/
        ├── neutral/
        ├── sad/
        └── surprise/
```

## Requirements

- Python 3.8 or higher
- Webcam
- Sufficient RAM (4GB+ recommended)
- GPU (optional, for faster training)

## Installation

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd emotion_detector
```

### 2. Create Virtual Environment

```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Quick Start (Using Pre-trained Model)

If you have a pre-trained `emotion_model.h5` file:

```bash
python main.py
```

### Training Your Own Model

1. **Prepare your dataset** in the following structure:
   ```
   data/
   ├── train/
   │   ├── angry/      # Training images for angry emotion
   │   ├── disgust/    # Training images for disgust emotion
   │   ├── fear/       # Training images for fear emotion
   │   ├── happy/      # Training images for happy emotion
   │   ├── neutral/    # Training images for neutral emotion
   │   ├── sad/        # Training images for sad emotion
   │   └── surprise/   # Training images for surprise emotion
   └── test/
       ├── angry/      # Test images for angry emotion
       ├── disgust/    # Test images for disgust emotion
       ├── fear/       # Test images for fear emotion
       ├── happy/      # Test images for happy emotion
       ├── neutral/    # Test images for neutral emotion
       ├── sad/        # Test images for sad emotion
       └── surprise/   # Test images for surprise emotion
   ```

2. **Train the model**:
   ```bash
   python train_emotion_model.py
   ```

3. **Run the emotion detector**:
   ```bash
   python main.py
   ```

## How It Works

1. **Face Detection**: Uses MediaPipe to detect faces in real-time from the webcam feed
2. **Image Preprocessing**: Extracts and resizes detected faces to 224x224 RGB format
3. **Emotion Prediction**: Feeds the processed face image through a MobileNetV2-based neural network
4. **Display**: Shows the webcam feed with bounding boxes around detected faces and predicted emotions

## Model Architecture

- **Base Model**: MobileNetV2 (pre-trained on ImageNet)
- **Input**: 224x224 RGB images
- **Output**: 7 emotion classes with softmax probabilities
- **Training**: Transfer learning with data augmentation and class balancing

## Controls

- **Press 'q'**: Quit the application
- **Camera window**: Shows real-time emotion detection

## Troubleshooting

### Camera Issues

- **"Could not find a working camera"**: 
  - Check camera permissions in system settings
  - Ensure no other application is using the camera
  - Try disconnecting and reconnecting the camera

- **"Failed to capture frame"**:
  - Wait a few seconds for camera initialization
  - Check camera drivers
  - Try a different camera if available

### Model Issues

- **"Model not found"**: 
  - Ensure `emotion_model.h5` exists in the project root
  - Train the model first using `train_emotion_model.py`

- **"Error predicting emotion"**:
  - Check if the model file is corrupted
  - Retrain the model if necessary

### Performance Issues

- **Slow performance**: 
  - Reduce image resolution in the code
  - Use a GPU for faster inference
  - Close other applications to free up resources

## Customization

### Adding New Emotions

1. Add new emotion folders to `data/train/` and `data/test/`
2. Update `EMOTION_LABELS` in `emotion_model.py`
3. Update `num_classes` in `train_emotion_model.py`
4. Retrain the model

### Adjusting Face Detection

Modify parameters in `face_detector.py`:
- `min_detection_confidence`: Lower for more sensitive detection
- `model_selection`: Use 1 for more accurate but slower detection

### Model Architecture Changes

Edit `train_emotion_model.py` to:
- Change the base model (e.g., ResNet, EfficientNet)
- Modify the classifier layers
- Adjust training parameters

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for face detection
- [TensorFlow](https://tensorflow.org/) for deep learning framework
- [OpenCV](https://opencv.org/) for computer vision operations
- [MobileNetV2](https://arxiv.org/abs/1801.04381) for the base model architecture 