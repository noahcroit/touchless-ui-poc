import argparse
import json
import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2



def flip_coordinate(frame, x, y):
    height, width, _ = frame.shape
    x_flip = width - x
    y_flip = y
    return x_flip, y_flip

def overlay_landmarker(frame, hand_result):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    l_gestures = hand_result.gestures
    l_hand_landmarks = hand_result.hand_landmarks
    l_handedness = hand_result.handedness
    l_text_gesture = []
    l_text_x = []
    l_text_y = []

    # Draw hand landmark
    for i in range(len(l_gestures)):
        gesture = l_gestures[i]
        hand_landmarks = l_hand_landmarks[i]
        handedness = l_handedness[i]

        #Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        mp_drawing.draw_landmarks(
                frame,
                hand_landmarks_proto,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        # Get the top left corner of the detected hand's bounding box.
        MARGIN = 20  # pixels
        height, width, _ = frame.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # flip the text coordinate
        text_x, text_y = flip_coordinate(frame, text_x, text_y)
        l_text_gesture.append(gesture[0].category_name)
        l_text_x.append(text_x)
        l_text_y.append(text_y)

    # Flip the image horizontally for a selfie-view display.
    frame = cv2.flip(frame, 1)

    # Draw text for gesture type
    FONT_SIZE = 1
    FONT_THICKNESS = 2
    TEXT_COLOR = (200, 240, 240)
    for g, x, y in zip(l_text_gesture, l_text_x, l_text_y):
        cv2.putText(frame, str(g), (x - 10, y + 50), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

    return frame



def handgesture_using_tasks(cam_url, model_path):
    BaseOptions = mp.tasks.BaseOptions
    GestureRecognizer = mp.tasks.vision.GestureRecognizer
    GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Create a gesture recognizer instance with the image mode:
    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=3,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5)
    recognizer = GestureRecognizer.create_from_options(options)

    # cv capture for webcam input
    # start capture
    cap = cv2.VideoCapture(cam_url)
    while cap.isOpened():
        # get frame
        ret, frame = cap.read()

        if not ret:
            print("Ignoring empty camera frame.")
            continue

        # pre-process frame
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        frame.flags.writeable = False
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform hand landmarks detection on the provided single image.
        # The hand landmarker must be created with the image mode.
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        detect_result = recognizer.recognize(mp_img)

        # print the category of
        gestures = detect_result.gestures
        handedness = detect_result.handedness
        for i in range(len(gestures)):
            print("handedness : {}, gesture type : {}".format(handedness[i][0].category_name, gestures[i][0].category_name))

        # Create overlay frame as frame_display
        frame_display = overlay_landmarker(frame, detect_result)

        # display the result
        frame_display = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
        cv2.imshow('MediaPipe Hands', frame_display)
        if cv2.waitKey(5) & 0xFF == 27:
            break
    cap.release()



if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser()
    # Adding optional argument
    parser.add_argument("-j", "--json", help="JSON file for the configuration", default='config.json')
    parser.add_argument("-t", "--usedtask", help="use task-level api or not (true, false)", default='true')

    # Read config file (for camera source, model etc)
    args = parser.parse_args()
    usedtask = args.usedtask
    f = open(args.json)
    data = json.load(f)
    cam = data['cam']
    model_path = data['model']['gesture']
    f.close()

    if usedtask:
        handgesture_using_tasks(cam, model_path)
