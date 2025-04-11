import cv2
import requests
import numpy as np
import argparse
import time
import os
import sys
from threading import Thread, Event

class Spinner:
    def __init__(self):
        self.spinner_chars = '|/-\\'
        self.stop_event = Event()

    def spin(self):
        i = 0
        while not self.stop_event.is_set():
            sys.stdout.write(f'\rProcessing... {self.spinner_chars[i]}')
            sys.stdout.flush()
            time.sleep(0.1)
            i = (i + 1) % 4
        sys.stdout.write('\rDone! \n')

class FaceClient:
    def __init__(self, server_url, show_boxes=False):
        self.server_url = server_url
        self.show_boxes = show_boxes

    def recognize(self):
        cap = cv2.VideoCapture(0)
        cv2.namedWindow('Face Recognition')
        captured_frame = None
        space_pressed = False

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            cv2.imshow('Face Recognition', frame)
            key = cv2.waitKey(1)

            if key == 32:  # Space pressed
                captured_frame = frame.copy()
                self._process_frame(captured_frame)

            if key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def _process_frame(self, frame):
        _, img_encoded = cv2.imencode('.jpg', frame)
        response = requests.post(
            f"{self.server_url}/recognize",
            files={'image': ('capture.jpg', img_encoded.tobytes(), 'image/jpeg')}
        )

        if response.ok:
            data = response.json()
            self._display_results(frame, data['faces'])

    def _display_results(self, frame, faces):
        print("\nRecognition Results:")
        for face in faces:
            confidence = 1 - face['distance']
            print(f"Identity: {face['name']} ({confidence:.2%})")
            if self.show_boxes:
                x, y, w, h = face['box']
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"{face['name']} ({confidence:.2%})",
                            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

def add_face(name, server_url):
    cap = cv2.VideoCapture(0)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Calculate the center of the frame
    center_x = frame_width // 2
    center_y = frame_height // 2

    # Define the bounding box size
    box_width = 400
    box_height = 250

    # Calculate the top-left corner of the bounding box
    target_box = [center_x - box_width // 2, center_y - box_height // 2, box_width, box_height]  # x, y, w, h

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    recording = False
    countdown = 0
    out = None

    cv2.namedWindow('Face Enrollment')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Draw the target rectangle
        cv2.rectangle(frame,
                      (target_box[0], target_box[1]),
                      (target_box[0]+target_box[2], target_box[1]+target_box[3]),
                      (0, 255, 0), 2)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        in_position = False
        for (x, y, w, h) in faces:
            if (x > target_box[0] and x+w < target_box[0]+target_box[2] and
                y > target_box[1] and y+h < target_box[1]+target_box[3]):
                in_position = True
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                break

        if recording:
            cv2.putText(frame, f"Recording: {round(countdown,2)}",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            out.write(frame)
            countdown -= 1/20  # 20 FPS
            if countdown <= 0:
                break
        else:
            cv2.putText(frame, "Position your face in the green box",
                        (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(frame, "Press SPACE to start recording",
                        (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow('Face Enrollment', frame)
        key = cv2.waitKey(1)

        if key == 32 and not recording and in_position:  # Start recording
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter('temp.mp4', fourcc, 20.0, (frame_width, frame_height))
            recording = True
            countdown = 3  # 3 seconds
        elif key == ord('q'):
            break

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

    if os.path.exists('temp.mp4'):
        spinner = Spinner()
        spinner_thread = Thread(target=spinner.spin)
        spinner_thread.start()
        try:
            with open('temp.mp4', 'rb') as f:
                response = requests.post(
                    f"{server_url}/add_face",
                    files={'video': f},
                    data={'name': name}
                )
                print(f"\n{response.json()['status']}")
        finally:
            spinner.stop_event.set()
            spinner_thread.join()
            os.remove('temp.mp4')

def list_faces(server_url):
    response = requests.get(f"{server_url}/list_faces")
    if response.ok:
        print("\nRegistered Faces:")
        for face in response.json()['faces']:
            print(f" - {face}")

def remove_face(name, server_url):
    response = requests.post(f"{server_url}/remove_face", data={'name': name})
    print(response.json()['status'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Face Recognition Client')
    subparsers = parser.add_subparsers(dest='command', required=True)

    add_parser = subparsers.add_parser('add')
    add_parser.add_argument('--name', required=True)

    rec_parser = subparsers.add_parser('recognize')
    rec_parser.add_argument('--show-boxes', action='store_true')

    subparsers.add_parser('list')

    remove_parser = subparsers.add_parser('remove')
    remove_parser.add_argument('--name', required=True)

    args = parser.parse_args()

    server_url = "http://localhost:5000"

    if args.command == 'add':
        add_face(args.name, server_url)
    elif args.command == 'recognize':
        FaceClient(server_url, args.show_boxes).recognize()
    elif args.command == 'list':
        list_faces(server_url)
    elif args.command == 'remove':
        remove_face(args.name, server_url)
