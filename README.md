# Gesture Recognition and Visual Effects

This Python script captures video from a camera, recognizes hand gestures using MediaPipe, and applies visual effects based on the recognized gestures. It leverages the OpenCV and MediaPipe libraries.

https://github.com/argrabowski/gesture-recognition/assets/64287065/27604ace-5f61-4395-8dd4-9ebf0dd3f60b

## Features

- Gesture recognition using MediaPipe's `vision.GestureRecognizer`.
- Applying visual effects such as overlaying images, drawing circles, lines, and applying color maps.
- Real-time video processing with OpenCV.

## Prerequisites

- Python 3.10
- OpenCV (`pip install opencv-python`)
- MediaPipe (`pip install mediapipe`)

## Getting Started

1. Clone the repository:

    ```bash
    git clone https://github.com/argrabowski/gesture-recognition.git
    cd gesture-recognition
    ```

2. Install the required packages:

    ```bash
    pip install opencv-python mediapipe
    ```

3. Run the script:

    ```bash
    python gesture_recognition.py
    ```

4. Press the `Esc` key to exit the application.
