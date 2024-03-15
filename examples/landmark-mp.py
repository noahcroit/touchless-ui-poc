import argparse
import json
import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2



def overlay_landmarker(frame, hand_result):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    l_hand_landmarks = hand_result.hand_landmarks
    l_handedness = hand_result.handedness

    for i in range(len(l_handedness)):
        #hand_landmarks = hand_landmarks_list[idx]
        #handedness = handedness_list[idx]
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

    return frame



def handlandmark_using_solutions(cam_url, model_path):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    hands = mp_hands.Hands(
        base_options=BaseOptions(model_asset_path=model_path),
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    # cv capture for webcam input
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

        # run hand process (detection + landmarking)
        hand_results = hands.process(frame)

        # display the result
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Hands', cv2.flip(frame, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()



def handlandmark_using_tasks(cam_url, model_path):
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Create a hand landmarker instance with the image mode:
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=3,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5)
    landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

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
        detect_result = landmarker.detect(mp_img)
        frame_display = overlay_landmarker(frame, detect_result)

        # Flip the image horizontally for a selfie-view display.
        frame_display = cv2.flip(frame_display, 1)

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
    model_path = data['model']['landmark']
    f.close()

    if usedtask:
        handlandmark_using_tasks(cam, model_path)
    else:
        handlandmark_using_solutions(cam, model_path)
