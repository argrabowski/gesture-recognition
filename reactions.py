import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# Apply overlay on frame
def apply_overlay(frame, image, alpha, positions):
    # Iterate over specified positions and blend image onto frame
    for start_y, end_y, start_x, end_x in positions:
        img_roi = frame[start_y:end_y, start_x:end_x]
        frame[start_y:end_y, start_x:end_x] = cv.addWeighted(img_roi, alpha, image, 1-alpha, 0)

    return frame

# Apply visual effects based on recognized gestures
def apply_effect(frame, gesture, images):
    # Get dimensions of frame
    frame_height, frame_width = frame.shape[:2]

    # Check recognized gesture and apply corresponding effect
    if gesture == 'Thumb_Up':
        # Thumb Up effect
        image = images['thumb_up']
        height, width = image.shape[:2]
        alpha = 0.3
        positions = [(height, 2 * height, width, 2 * width)]
        color_map = cv.COLORMAP_DEEPGREEN

        # Draw circles on frame
        cv.circle(frame, (200, 200), 35, (200, 200, 200), 2)
        cv.circle(frame, (250, 240), 15, (200, 200, 200), 2)
    elif gesture == 'Thumb_Down':
        # Thumb Down effect
        image = images['thumb_down']
        height, width = image.shape[:2]
        alpha = 0.3
        positions = [(height, 2 * height, width, 2 * width)]
        color_map = cv.COLORMAP_INFERNO

        # Draw circles on frame
        cv.circle(frame, (200, 200), 35, (200, 200, 200), 2)
        cv.circle(frame, (250, 240), 15, (200, 200, 200), 2)
    elif gesture == 'Victory':
        # Peace sign effect
        image = images['balloon']
        height, width = image.shape[:2]
        alpha = 0.3
        positions = [
            (0, height, 0, width),
            (0, height, frame_width-width, frame_width)]
        color_map = cv.COLORMAP_PARULA

        # Draw circles on frame
        num_circles = 20
        for i in range(num_circles):
            radius = 20 + i * 20
            center = (frame_width // 2, frame_height // 2)
            cv.circle(frame, center, radius, (0, 255, 0), 2)
    elif gesture == 'ILoveYou':
        # Horns sign effect
        image = images['explosion']
        height, width = image.shape[:2]
        alpha = 0.3
        positions = [
            (0, height, 0, width),
            (0, height, frame_width-width, frame_width)]
        color_map = cv.COLORMAP_RAINBOW

        # Draw lines on frame
        num_lines = 5
        line_y_coordinates = np.linspace(50, 350, num_lines)
        for y in line_y_coordinates:
            points = np.array([
                [50, y], [100, y + 50], [150, y], [200, y + 50],
                [250, y], [300, y + 50], [350, y], [400, y + 50],
                [450, y], [500, y + 50], [550, y], [600, y + 50]], np.int32)
            points = points.reshape((-1, 1, 2))
            cv.polylines(frame, [points], isClosed=False, color=(0, 0, 255), thickness=2)
    elif gesture == 'Closed_Fist':
        # Closed fist effect
        image = images['sad_face']
        height, width = image.shape[:2]
        alpha = 0.3
        positions = [(2 * height, 3 * height, 3 * width, 4 * width)]
        color_map = cv.COLORMAP_BONE
    elif gesture == 'Open_Palm':
        # Open palm effect
        image = images['happy_face']
        height, width = image.shape[:2]
        alpha = 0.3
        positions = [(2 * height, 3 * height, 3 * width, 4 * width)]
        color_map = cv.COLORMAP_PLASMA
    else:
        # Unrecognized gesture
        return frame

    # Apply color map to entire frame
    frame = cv.applyColorMap(frame, color_map)

    # Apply overlay on frame with specified image at given positions
    apply_overlay(frame, image, alpha, positions)

    return frame

# Perform gesture recognition and apply effects
def main():
    # Set up gesture recognizer using MediaPipe
    base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
    options = vision.GestureRecognizerOptions(base_options=base_options)
    recognizer = vision.GestureRecognizer.create_from_options(options)

    # Load images for different effects
    images = {
        'thumb_up': cv.imread('thumb_up.png'),
        'thumb_down': cv.imread('thumb_down.png'),
        'balloon': cv.imread('balloon.png'),
        'explosion': cv.imread('explosion.png'),
        'happy_face': cv.imread('happy_face.png'),
        'sad_face': cv.imread('sad_face.png')}

    # Open video capture
    cap = cv.VideoCapture(0)

    # Capture video frames and apply effects
    while cap.isOpened():
        # Read frame from video capture
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB format for MediaPipe
        rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv.cvtColor(frame, cv.COLOR_BGR2RGB))

        # Recognize gestures using gesture recognizer
        results = recognizer.recognize(rgb_frame)

        # If gestures recognized, apply effect to frame
        if results.gestures:
            top_result = results.gestures[0][0].category_name
            frame = apply_effect(frame, top_result, images)

        # Display frame with applied effects
        cv.imshow('Reactions', frame)

        # Exit if Esc key pressed
        key = cv.waitKey(1)
        if key == 27:
            break

    # Release video capture and close windows
    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
