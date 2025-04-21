

import cv2
import requests
import numpy as np
import argparse
import time
import os
import sys
from threading import Thread, Event
import json
import webbrowser
import tempfile


class Spinner:
    def __init__(self, message="Processing..."):
        self.spinner_chars = '|/-\\'
        self.stop_event = Event()
        self.message = message
        self.thread = None

    def _spin(self):
        i = 0
        while not self.stop_event.is_set():
            sys.stdout.write(f'\r{self.message} {self.spinner_chars[i]}')
            sys.stdout.flush()
            time.sleep(0.1)
            i = (i + 1) % len(self.spinner_chars)
        sys.stdout.write('\r' + ' ' * (len(self.message) + 2) + '\r')
        sys.stdout.flush()

    def start(self):
        self.stop_event.clear()
        self.thread = Thread(target=self._spin, daemon=True)
        self.thread.start()

    def stop(self):
        if self.thread and self.thread.is_alive():
            self.stop_event.set()
            self.thread.join(timeout=0.5)


class FaceClient:
    def __init__(self, server_url, show_boxes=False):
        self.server_url = server_url.rstrip('/')
        self.show_boxes = show_boxes
        self.active_server_model = "Unknown"

    def _make_request(self, method, endpoint, expect_json=True, **kwargs):

        url = f"{self.server_url}/{endpoint}"

        headers = kwargs.pop('headers', {'Accept': 'application/json'})
        if not expect_json:
            headers['Accept'] = 'text/html'

        try:
            response = requests.request(
                method, url, timeout=120, headers=headers, **kwargs)
            response.raise_for_status()
            if expect_json:
                return response.json()
            else:

                return response.text
        except requests.exceptions.ConnectionError:
            print(
                f"\nError: Could not connect to the server at {self.server_url}.", file=sys.stderr)
            sys.exit(1)
        except requests.exceptions.Timeout:
            print(
                f"\nError: Request timed out connecting to {url}.", file=sys.stderr)
            return None
        except requests.exceptions.RequestException as e:
            print(f"\nError during request to {url}: {e}", file=sys.stderr)
            if hasattr(e, 'response') and e.response is not None:

                try:
                    print(
                        f"Server response: {e.response.json()}", file=sys.stderr)
                except ValueError:
                    print(
                        f"Server response ({e.response.status_code}): {e.response.text}", file=sys.stderr)
            return None
        except json.JSONDecodeError as e:
            if expect_json:
                print(
                    f"\nError: Could not decode JSON response from {url}. Response text: {response.text}", file=sys.stderr)
                return None
            else:
                return response.text

    def recognize_webcam(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.", file=sys.stderr)
            return
        window_name = 'Face Recognition (Press SPACE to capture, Q to quit)'
        cv2.namedWindow(window_name)
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame from webcam.", file=sys.stderr)
                break
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 32:
                print("\nCapturing frame...")
                spinner = Spinner("Sending frame and recognizing...")
                spinner.start()
                try:
                    _, img_encoded = cv2.imencode('.jpg', frame)
                    files = {
                        'image': ('capture.jpg', img_encoded.tobytes(), 'image/jpeg')}
                    response_data = self._make_request(
                        'post', 'recognize', files=files)
                finally:
                    spinner.stop()
                if response_data:
                    print("Recognition Results:")
                    if response_data.get("faces"):
                        self._display_results(
                            frame.copy(), response_data['faces'])
                    else:
                        print(
                            " - No faces detected or recognized in the captured frame.")
                else:
                    print("Recognition failed. Check server connection and logs.")
                cv2.waitKey(0)
            elif key == ord('q'):
                print("Exiting webcam recognition.")
                break
        cap.release()
        cv2.destroyAllWindows()

    def recognize_file(self, image_path):
        if not os.path.exists(image_path):
            print(
                f"Error: Image file not found at '{image_path}'", file=sys.stderr)
            return
        print(f"Recognizing faces in file: {image_path}")
        spinner = Spinner("Uploading and recognizing...")
        spinner.start()
        try:
            with open(image_path, 'rb') as f:
                files = {'image': (os.path.basename(
                    image_path), f, 'image/jpeg')}
                response_data = self._make_request(
                    'post', 'recognize', files=files)
        finally:
            spinner.stop()
        if response_data:
            print("\nRecognition Results:")
            faces = response_data.get("faces")
            if faces:
                img_display = cv2.imread(
                    image_path) if self.show_boxes else None
                for face in faces:
                    name = face.get('name', 'error')
                    confidence = float(
                        face.get('confidence', 0.0) or 0.0) * 100
                    distance = face.get('distance')
                    distance_str = f"{distance:.4f}" if distance is not None else "N/A"
                    print(
                        f" - Found: {name} (Confidence: {confidence:.2f}%, Distance: {distance_str})")
                    if self.show_boxes and img_display is not None and name != 'error':
                        x, y, w, h = face['box']
                        color = (0, 0, 255) if name == "unknown" else (
                            0, 255, 0)
                        cv2.rectangle(img_display, (x, y),
                                      (x+w, y+h), color, 2)
                        text = f"{name} ({confidence:.1f}%)"
                        cv2.putText(img_display, text, (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                if self.show_boxes and img_display is not None:
                    cv2.imshow(
                        f'Recognition Results: {os.path.basename(image_path)}', img_display)
                    print("\nPress any key in the image window to close it.")
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
            else:
                print(" - No faces detected or recognized in the image.")
        else:
            print("Recognition failed. Check server connection and logs.")

    def _display_results(self, frame, faces):
        if not faces:
            print(" - No faces returned by server.")
            return
        display_frame = frame.copy()
        for face in faces:
            name = face.get('name', 'error')
            confidence = float(face.get('confidence', 0.0) or 0.0) * 100
            distance = face.get('distance')
            distance_str = f"{distance:.4f}" if distance is not None else "N/A"
            print(
                f" - Found: {name} (Confidence: {confidence:.2f}%, Distance: {distance_str})")
            if self.show_boxes and name != 'error':
                x, y, w, h = face['box']
                color = (0, 0, 255) if name == "unknown" else (0, 255, 0)
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                text = f"{name} ({confidence:.1f}%)"
                cv2.putText(display_frame, text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if self.show_boxes:
            cv2.imshow(
                'Recognition Result (Press any key to continue)', display_frame)

    def add_face_webcam(self, name):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.", file=sys.stderr)
            return
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        box_width = int(frame_width * 0.6)
        box_height = int(frame_height * 0.6)
        target_x = (frame_width - box_width) // 2
        target_y = (frame_height - box_height) // 2
        target_box = [target_x, target_y, box_width, box_height]
        face_cascade_path = os.path.join(
            cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
        if not os.path.exists(face_cascade_path):
            print(
                f"Error: Haar cascade file not found at {face_cascade_path}", file=sys.stderr)
            cap.release()
            return
        face_cascade = cv2.CascadeClassifier(face_cascade_path)
        recording = False
        countdown = 0
        out = None
        video_filename = f"temp_{name}_{int(time.time())}.mp4"
        record_duration_sec = 5
        fps = 20.0
        cv2.namedWindow('Face Enrollment (SPACE to start, Q to quit)')
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.", file=sys.stderr)
                break
            display_frame = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            in_position = False
            cv2.rectangle(display_frame, (target_box[0], target_box[1]), (
                target_box[0]+target_box[2], target_box[1]+target_box[3]), (0, 255, 0), 2)
            for (x, y, w, h) in faces:
                face_center_x = x + w // 2
                face_center_y = y + h // 2
                if (target_box[0] < face_center_x < target_box[0] + target_box[2] and target_box[1] < face_center_y < target_box[1] + target_box[3]):
                    cv2.rectangle(display_frame, (x, y),
                                  (x+w, y+h), (255, 0, 0), 2)
                    if h > target_box[3] / 4:
                        in_position = True
                        break
            if recording:
                record_time_left = max(0, countdown)
                cv2.putText(display_frame, f"RECORDING: {record_time_left:.1f}s", (
                    target_box[0], target_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if out:
                    out.write(frame)
                    countdown -= 1 / fps
                if countdown <= 0:
                    print("\nRecording finished.")
                    break
            elif in_position:
                cv2.putText(display_frame, "Face detected in position.", (
                    target_box[0], target_box[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, "Press SPACE to start recording", (
                    target_box[0], target_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                cv2.putText(display_frame, "Position your face inside the green box", (
                    target_box[0], target_box[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_frame, "(Ensure good lighting and clear view)", (
                    target_box[0], target_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow(
                'Face Enrollment (SPACE to start, Q to quit)', display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 32 and not recording and in_position:
                print("\nStarting recording...")
                video_filepath = video_filename
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(
                    video_filepath, fourcc, fps, (frame_width, frame_height))
                if not out.isOpened():
                    print(
                        f"Error: Could not open video writer for {video_filepath}", file=sys.stderr)
                    break
                recording = True
                countdown = record_duration_sec
            elif key == ord('q'):
                print("\nEnrollment cancelled.")
                if recording and out:
                    out.release()
                    out = None
                if os.path.exists(video_filepath):
                    os.remove(video_filepath)
                    print(f"Deleted incomplete recording: {video_filepath}")
                cap.release()
                cv2.destroyAllWindows()
                return
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        if os.path.exists(video_filepath) and recording:
            print(f"Video saved: {video_filepath}")
            self.add_face_file(name, video_filepath)
            try:
                os.remove(video_filepath)
                print(f"Removed temporary video file: {video_filepath}")
            except OSError as e:
                print(
                    f"Warning: Could not remove temporary video file {video_filepath}: {e}", file=sys.stderr)
        elif not recording:
            print("Recording was not started or completed.")

    def add_face_file(self, name, video_path):
        if not os.path.exists(video_path):
            print(
                f"Error: Video file not found at '{video_path}'", file=sys.stderr)
            return
        print(f"\nAdding face for '{name}' using video: {video_path}")
        spinner = Spinner("Uploading video and processing on server...")
        spinner.start()
        try:
            with open(video_path, 'rb') as f:
                files = {'video': (os.path.basename(
                    video_path), f, 'video/mp4')}
                data = {'name': name}
                response_data = self._make_request(
                    'post', 'add_face', files=files, data=data)
        finally:
            spinner.stop()
        if response_data:
            status = response_data.get('status', 'No status message')
            count = response_data.get('count', 0)
            print(f"Server Response: {status}")
            if count > 0:
                print(f"Successfully added {count} embeddings.")
            else:
                print(
                    "No new embeddings were added. Check video quality or server logs.")
        else:
            print("Failed to add face. Check server connection and logs.")

    def list_faces(self):
        print("\nRequesting list of registered faces from server...")
        spinner = Spinner("Fetching data...")
        spinner.start()
        response_data = self._make_request('get', 'list_faces')
        spinner.stop()
        if response_data:
            names = response_data.get('registered_names', [])
            model = response_data.get('active_model', 'N/A')
            collection = response_data.get('collection_name', 'N/A')
            count = response_data.get('total_embeddings', 0)
            print(f"\n--- Server Status ---")
            print(f"Active Model: {model}")
            print(f"Collection Name: {collection}")
            print(f"Total Embeddings in Collection: {count}")
            print(f"--- Registered Names ({len(names)}) ---")
            if names:
                for name in names:
                    print(f" - {name}")
            else:
                print("No faces registered yet.")
            print("--------------------")
            self.active_server_model = model
        else:
            print("Failed to retrieve list of faces.")

    def remove_face_by_name(self, name):

        print(f"\nRequesting removal of all entries for name: '{name}'...")
        spinner = Spinner("Sending removal request...")
        spinner.start()
        data = {'name': name}

        response_data = self._make_request('post', 'remove_face', data=data)
        spinner.stop()
        if response_data:
            print(
                f"Server Response: {response_data.get('status', 'No status message')}")
        else:
            print(f"Failed to remove face '{name}'.")

    def remove_face_by_id(self, ids_to_remove):

        if not ids_to_remove:
            print("Error: No IDs provided to remove.", file=sys.stderr)
            return

        print(f"\nRequesting removal of specific IDs: {ids_to_remove}...")
        spinner = Spinner("Sending removal request...")
        spinner.start()

        payload = {'ids': ids_to_remove}

        response_data = self._make_request(
            'post', 'remove_id', expect_json=True, json=payload)
        spinner.stop()

        if response_data:
            print(
                f"Server Response: {response_data.get('status', 'No status message')}")
            removed = response_data.get('removed_ids', [])
            not_found = response_data.get('not_found_ids', [])
            if removed:
                print(f" - Successfully removed: {removed}")
            if not_found:
                print(f" - IDs not found: {not_found}")
        else:
            print(f"Failed to remove faces by ID.")

    def visualize_embeddings(self, reduction_method='tsne'):

        print(
            f"\nRequesting {reduction_method.upper()} visualization from server...")
        spinner = Spinner(f"Generating {reduction_method} visualization...")
        spinner.start()

        html_content = self._make_request(
            'get', f'visualize?reduction={reduction_method}', expect_json=False)
        spinner.stop()

        if html_content and isinstance(html_content, str) and html_content.strip().startswith('<'):
            try:

                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.html', encoding='utf-8') as tmp_file:
                    tmp_file.write(html_content)
                    tmp_filepath = tmp_file.name
                print(f"Visualization saved temporarily to: {tmp_filepath}")

                print("Opening visualization in your web browser...")
                webbrowser.open(f'file://{os.path.realpath(tmp_filepath)}')

                input(
                    "Press Enter to close this script (the browser window will remain open).")

            except Exception as e:
                print(
                    f"\nError saving or opening visualization: {e}", file=sys.stderr)

        elif html_content:
            print("Server sent back an HTML error page or unexpected content.")

            try:
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='_error.html', encoding='utf-8') as tmp_file:
                    tmp_file.write(html_content)
                    tmp_filepath = tmp_file.name
                print(f"Server error page saved to: {tmp_filepath}")
                webbrowser.open(f'file://{os.path.realpath(tmp_filepath)}')
                input("Press Enter to close.")
            except:
                pass
        else:
            print("Failed to retrieve visualization from server. Check server logs.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Face Recognition Client', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--server', default="http://localhost:5000",
                        help="URL of the face recognition server (default: http://localhost:5000)")
    subparsers = parser.add_subparsers(
        dest='command', required=True, help='Available commands')

    add_parser = subparsers.add_parser(
        'add', help='Add a new face identity via webcam or video file.')
    add_parser.add_argument('--name', required=True,
                            help='Name of the person to add.')
    add_parser.add_argument(
        '--file', help='Path to a video file (.mp4, etc.) for enrollment instead of webcam.')

    rec_parser = subparsers.add_parser(
        'recognize', help='Recognize faces via webcam or image file.')
    rec_parser.add_argument(
        '--file', help='Path to an image file (.jpg, .png) to recognize faces in.')
    rec_parser.add_argument('--show-boxes', action='store_true',
                            help='Display image/webcam feed with bounding boxes.')

    list_parser = subparsers.add_parser(
        'list', help='List all registered face names and server info.')

    remove_name_parser = subparsers.add_parser(
        'remove', help='Remove ALL entries for a specific face name.')
    remove_name_parser.add_argument(
        '--name', required=True, help='Name of the person to remove.')

    remove_id_parser = subparsers.add_parser(
        'remove-id', help='Remove specific face entries by their IDs.')
    remove_id_parser.add_argument(
        '--ids', required=True, nargs='+', help='One or more space-separated IDs to remove.')

    viz_parser = subparsers.add_parser(
        'visualize', help='Generate and view embedding visualization.')
    viz_parser.add_argument('--reduction', default='tsne', choices=[
                            'tsne', 'umap'], help='Dimensionality reduction method (default: tsne).')

    args = parser.parse_args()
    client = FaceClient(args.server, getattr(args, 'show_boxes', False))

    if args.command == 'add':
        if args.file:
            client.add_face_file(args.name, args.file)
        else:
            client.add_face_webcam(args.name)
    elif args.command == 'recognize':
        if args.file:
            client.recognize_file(args.file)
        else:
            client.recognize_webcam()
    elif args.command == 'list':
        client.list_faces()
    elif args.command == 'remove':
        client.remove_face_by_name(args.name)
    elif args.command == 'remove-id':
        client.remove_face_by_id(args.ids)
    elif args.command == 'visualize':
        client.visualize_embeddings(args.reduction)
    else:
        parser.print_help()
