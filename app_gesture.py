import argparse
import sys
import json
import cv2
from gesture_controller import GestureController
import threading
import time
import redis
import queue



# Queue
q_pos = queue.Queue()
q_pos_confirm = queue.Queue()
istaskrun_cv = False
istaskrun_redis = False

def task_cv(cam_url, model_path):

    # Hand Gesture Ctrl
    g = GestureController(5, 5, control_hand='Right')
    g.config(model_path)

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
        pos, frame_display = g.step(frame)
        print(pos)
        print("x pos:{}, y pos:{}".format(g.x, g.y))
        if pos:
            q_pos_confirm.put(pos)
        if g.state == 'ACTIVE':
            q_pos.put((g.x, g.y))

        # display the result
        frame_display = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
        cv2.imshow('MediaPipe Hands', frame_display)
        if cv2.waitKey(100) & 0xFF == 27:
            break
    cap.release()
    istaskrun_cv = False




def task_redis():
    # redis client
    r = redis.Redis(host='localhost', port=6379)
    # redis channels for tags
    ch_pos_x = 'handgesture.pos_x'
    ch_pos_y = 'handgesture.pos_y'
    ch_pos_confirm_x = 'handgesture.pos_confirm_x'
    ch_pos_confirm_y = 'handgesture.pos_confirm_y'
    ch_state = 'handgesture.state'

    while istaskrun_cv:
        # tag for cursor position
        if not q_pos.empty():
            x, y = q_pos.get()
            r.set(ch_pos_x, x)
            r.set(ch_pos_y, y)

        if not q_pos_confirm.empty():
            x, y = q_pos_confirm.get()
            r.set(ch_pos_confirm_x, x)
            r.set(ch_pos_confirm_y, y)

        time.sleep(0.1)




if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser()
    # Adding optional argument
    parser.add_argument("-j", "--json", help="JSON file for the configuration", default='config.json')

    # Read config file (for camera source, model etc)
    args = parser.parse_args()
    f = open(args.json)
    data = json.load(f)
    cam = data['cam']
    model_path = data['model']['gesture']
    f.close()

    # Run CV task
    istaskrun_cv = True
    istaskrun_redis = True
    t1 = threading.Thread(target=task_cv, args=(cam, model_path))
    t2 = threading.Thread(target=task_redis)
    t1.start()
    t2.start()
    t1.join()
    t2.join()


