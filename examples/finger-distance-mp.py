import argparse
import json
import math
import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2



def flip_coordinate(frame, x, y):
    height, width, _ = frame.shape
    x_flip = width - x
    y_flip = y
    return x_flip, y_flip

def denormalize_landmark(frame, x, y):
    height, width, _ = frame.shape
    x_denom = int(x*width)
    y_denom = int(y*height)
    return x_denom, y_denom

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

        # Draw the hand landmarks.
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
    for g, x, y in zip(l_text_gesture, l_text_x, l_text_y):
        cv2.putText(frame, str(g), (x - 10, y + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (240, 240, 240), 1)

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

        # get thumb-to-index distance
        distances = []
        l_hand_landmarks = detect_result.hand_landmarks
        for hand_landmarks in l_hand_landmarks:
            x_thumb = hand_landmarks[4].x
            x_index = hand_landmarks[8].x
            y_thumb = hand_landmarks[4].y
            y_index = hand_landmarks[8].y
            d = math.dist((x_thumb, y_thumb), (x_index, y_index))
            distances.append(d)

            # draw point & line at the tip of thumb & index finger
            # denormalize first
            x_thumb, y_thumb = denormalize_landmark(frame_display, x_thumb, y_thumb)
            x_index, y_index = denormalize_landmark(frame_display, x_index, y_index)
            x_thumb, y_thumb = flip_coordinate(frame, x_thumb, y_thumb)
            x_index, y_index = flip_coordinate(frame, x_index, y_index)
            print(x_thumb, y_thumb)
            print(x_index, y_index)
            cv2.circle(frame_display, (x_thumb, y_thumb), 20, (0, 0, 255), -1)
            cv2.circle(frame_display, (x_index, y_index), 20, (0, 0, 255), -1)
            cv2.line(frame_display, (x_thumb, y_thumb), (x_index, y_index), (255, 255, 0), 5)
            cv2.putText(frame_display, "%.3f" % d, (x_index - 10, y_index - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 0), 2)
        print("distance : ", distances)

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
