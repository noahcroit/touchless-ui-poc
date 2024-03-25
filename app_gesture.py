import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2


HANDPOSE_START = 'Closed_Fist'
HANDPOSE_CONFIRM = 'Victory'
HANDPOSE_CANCEL = 'Open_Palm'
HANDPOSE_UP = 'Thumb_Up'
HANDPOSE_DOWN = 'Thumb_Down'
HANDPOSE_RIGHT = 'Pointing_Up'
HANDPOSE_LEFT = 7


class GestureController:
    def __init__(self, size_x, size_y, display=False, selfie=True, control_hand='Right'):
        self.display_flag = display
        self.selfie = selfie
        self.state = 'IDLE'
        self.x = 0
        self.y = 0
        self.size_x = size_x
        self.size_y = size_y
        self.control_hand = control_hand
        self.detect_result = None
        self.detect_frame = None

    def config(self, model_path):
        # Create a gesture recognizer instance with the image mode:
        BaseOptions = mp.tasks.BaseOptions
        GestureRecognizer = mp.tasks.vision.GestureRecognizer
        GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=3,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        self.recognizer = GestureRecognizer.create_from_options(options)

    def getHandGesture(self, frame):
        self.detect_frame = frame
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        detect_result = self.recognizer.recognize(mp_img)
        self.detect_result = detect_result
        gestures = detect_result.gestures
        handedness = detect_result.handedness
        for i in range(len(handedness)):
            if handedness[i][0].category_name == self.control_hand:
                return gestures[i][0].category_name
        return None

    def flip_coordinate(self, frame, x, y):
        height, width, _ = frame.shape
        x_flip = width - x
        y_flip = y
        return x_flip, y_flip

    def overlayFrame(self):
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_hands = mp.solutions.hands

        l_gestures = self.detect_result.gestures
        l_hand_landmarks = self.detect_result.hand_landmarks
        l_handedness = self.detect_result.handedness
        l_text_gesture = []
        l_text_x = []
        l_text_y = []
        frame = self.detect_frame

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

            if self.selfie:
                # flip the text coordinate
                text_x, text_y = self.flip_coordinate(frame, text_x, text_y)

            l_text_gesture.append(gesture[0].category_name)
            l_text_x.append(text_x)
            l_text_y.append(text_y)

        if self.selfie:
            # Flip the image horizontally for a selfie-view display.
            frame = cv2.flip(frame, 1)

        # Draw text for gesture type
        FONT_SIZE = 1
        FONT_THICKNESS = 2
        TEXT_COLOR = (200, 240, 240)
        for g, x, y in zip(l_text_gesture, l_text_x, l_text_y):
            cv2.putText(frame, str(g), (x - 10, y + 50), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

        return frame

    def step(self, frame):
        pos = None
        # get hand pose
        gesture = self.getHandGesture(frame)

        # if gesture is detected
        if not gesture:
            if self.selfie:
                # Flip the image horizontally for a selfie-view display.
                frame = cv2.flip(frame, 1)
            return pos, frame
        else:
            frame = self.overlayFrame()

        if self.state == 'IDLE':
            if gesture == HANDPOSE_START:
                self.state = 'ACTIVE'
        elif self.state == 'ACTIVE':
            if gesture == HANDPOSE_CONFIRM:
                self.state = 'IDLE'
                pos = (self.x, self.y)
            elif gesture == HANDPOSE_CANCEL:
                self.state = 'IDLE'
            else:
                # move x, y depended on the hand pose (up, down, right, left)
                if gesture == HANDPOSE_UP:
                    self.y += 1
                    if self.y >= self.size_y:
                        self.y = 0
                elif gesture == HANDPOSE_DOWN:
                    self.y -= 1
                    if self.y < 0:
                        self.y = self.size_y - 1
                elif gesture == HANDPOSE_RIGHT:
                    self.x += 1
                    if self.x >= self.size_x:
                        self.x = 0
                elif gesture == HANDPOSE_LEFT:
                    self.x -= 1
                    if self.x < 0:
                        self.x = self.size_x - 1

        return pos, frame
