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
import base64 


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
        if not sys.stdout.isatty(): 
             print(f"{self.message} (running in non-interactive mode)")
             return
        self.stop_event.clear()
        self.thread = Thread(target=self._spin, daemon=True)
        self.thread.start()

    def stop(self):
        if self.thread and self.thread.is_alive():
            self.stop_event.set()
            try:
                self.thread.join(timeout=0.5) 
            except Exception:
                 pass 
        
        if sys.stdout.isatty():
             sys.stdout.write('\r' + ' ' * (len(self.message) + 2) + '\r')
             sys.stdout.flush()



class FaceClient:
    def __init__(self, server_url, show_boxes=False):
        self.server_url = server_url.rstrip('/')
        self.show_boxes = show_boxes
        self.active_server_model = "Unknown" 

    def _make_request(self, method, endpoint, expect_json=True, **kwargs):
        
        url = f"{self.server_url}/{endpoint}"
        headers = kwargs.pop('headers', {})

        
        if expect_json and 'Accept' not in headers:
            headers['Accept'] = 'application/json'
        elif not expect_json and 'Accept' not in headers:
             
             headers['Accept'] = 'text/html, text/plain, */*'

        
        timeout = kwargs.pop('timeout', 120)

        try:
            response = requests.request(method, url, timeout=timeout, headers=headers, **kwargs)

            
            response.raise_for_status()

            
            if expect_json:
                try:
                    return response.json()
                except json.JSONDecodeError as e:
                    print(f"\nError: Could not decode JSON response from {url}.", file=sys.stderr)
                    print(f"Status Code: {response.status_code}", file=sys.stderr)
                    print(f"Response Text: {response.text[:500]}...", file=sys.stderr) 
                    return None 
            else:
                
                return response.text

        except requests.exceptions.ConnectionError:
            print(f"\nError: Could not connect to the server at {self.server_url}.", file=sys.stderr)
            print("Please ensure the server is running and accessible.", file=sys.stderr)
            sys.exit(1) 
        except requests.exceptions.Timeout:
            print(f"\nError: Request timed out connecting to {url} (timeout={timeout}s).", file=sys.stderr)
            print("The server might be busy or unresponsive. Try increasing the timeout or check server load.", file=sys.stderr)
            return None 
        except requests.exceptions.HTTPError as e:
             print(f"\nError: HTTP Error {e.response.status_code} for URL {url}.", file=sys.stderr)
             
             try:
                 error_data = e.response.json()
                 print(f"Server Error Message: {error_data.get('error', 'No error detail provided.')}", file=sys.stderr)
             except json.JSONDecodeError:
                 print(f"Server Response (text): {e.response.text[:500]}...", file=sys.stderr)
             return None 
        except requests.exceptions.RequestException as e:
            
            print(f"\nError: An unexpected error occurred during the request to {url}: {e}", file=sys.stderr)
            return None 


    def recognize_webcam(self):
        
        cap = cv2.VideoCapture(0) 
        if not cap.isOpened():
            print("Error: Could not open webcam. Check permissions and device connection.", file=sys.stderr)
            return

        window_name = 'Face Recognition (SPACE to Capture, Q to Quit)'
        cv2.namedWindow(window_name)
        print("\nStarting webcam recognition...")
        print("Press SPACE bar to capture and recognize the current frame.")
        print("Press 'q' to quit.")

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
                response_data = None
                try:
                    
                    ret, img_encoded = cv2.imencode('.jpg', frame)
                    if not ret:
                         print("Error: Failed to encode frame as JPEG.", file=sys.stderr)
                         continue

                    files = {'image': ('capture.jpg', img_encoded.tobytes(), 'image/jpeg')}

                    
                    response_data = self._make_request('post', 'recognize', files=files, expect_json=True)

                finally:
                    spinner.stop() 

                if response_data:
                    print("Recognition Results:")
                    faces = response_data.get('faces')
                    if faces:
                         self._display_results(frame.copy(), faces) 
                    else:
                         
                         print(" - No faces detected or recognized in the captured frame.")
                         if self.show_boxes: 
                             cv2.imshow('Recognition Result (Press any key to continue)', frame)
                             cv2.waitKey(0) 
                else:
                    print("Recognition failed. Check server connection and logs.")
                    
                    cv2.waitKey(1) 

            elif key == ord('q'):
                print("\nExiting webcam recognition.")
                break

        
        cap.release()
        cv2.destroyAllWindows()


    def recognize_file(self, image_path):
        
        if not os.path.exists(image_path):
            print(f"Error: Image file not found at '{image_path}'", file=sys.stderr)
            return

        print(f"\nRecognizing faces in file: {image_path}")
        spinner = Spinner("Uploading image and recognizing...")
        spinner.start()
        response_data = None
        try:
            with open(image_path, 'rb') as f:
                files = {'image': (os.path.basename(image_path), f, 'image/jpeg')} 
                response_data = self._make_request('post', 'recognize', files=files, expect_json=True)
        except IOError as e:
             print(f"Error opening image file {image_path}: {e}", file=sys.stderr)
             spinner.stop()
             return
        finally:
            spinner.stop()

        if response_data:
            print("\nRecognition Results:")
            faces = response_data.get("faces") 

            if faces is None:
                 print("Error: Server response did not contain 'faces' key.")
                 return

            img_display = None
            if self.show_boxes:
                try:
                    img_display = cv2.imread(image_path)
                    if img_display is None:
                        print(f"Warning: Could not read image {image_path} for displaying boxes.", file=sys.stderr)
                except Exception as e:
                    print(f"Warning: Error reading image {image_path} for display: {e}", file=sys.stderr)
                    img_display = None


            if faces: 
                print(f"Found {len(faces)} face(s) meeting criteria:")
                for i, face in enumerate(faces):
                    name = face.get('name', 'error')
                    confidence = float(face.get('confidence', 0.0) or 0.0) * 100 
                    distance = face.get('distance') 
                    distance_str = f"{distance:.4f}" if distance is not None else "N/A"
                    box_str = str(face.get('box', 'N/A'))

                    print(f" - Face {i+1}: Name='{name}', Confidence={confidence:.2f}%, Distance={distance_str}, Box={box_str}")

                    
                    if self.show_boxes and img_display is not None and name != 'error' and 'box' in face:
                        try:
                            x, y, w, h = map(int, face['box'])
                            
                            if name == "unknown": color = (0, 165, 255) 
                            elif name.startswith("error"): color = (0, 0, 255) 
                            else: color = (0, 255, 0) 

                            cv2.rectangle(img_display, (x, y), (x + w, y + h), color, 2)
                            
                            text = f"{name} ({confidence:.1f}%)"
                            text_y = y - 10 if y - 10 > 10 else y + h + 20
                            
                            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                            cv2.rectangle(img_display, (x, text_y - th - 5), (x + tw, text_y), color, -1)
                            cv2.putText(img_display, text, (x, text_y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2) 
                        except Exception as draw_e:
                             print(f"Warning: Error drawing box/text for face {i+1}: {draw_e}", file=sys.stderr)

                
                if self.show_boxes and img_display is not None:
                    try:
                        window_title = f'Recognition Results: {os.path.basename(image_path)} (Press any key)'
                        cv2.imshow(window_title, img_display)
                        print("\nPress any key in the image window to close it.")
                        cv2.waitKey(0)
                        cv2.destroyWindow(window_title) 
                    except Exception as display_e:
                        print(f"Error displaying image: {display_e}", file=sys.stderr)
                        cv2.destroyAllWindows() 

            else: 
                print(" - No faces detected or recognized in the image that meet the criteria.")
                
                if self.show_boxes and img_display is not None:
                     try:
                         window_title = f'No Faces Found: {os.path.basename(image_path)} (Press any key)'
                         cv2.imshow(window_title, img_display)
                         print("\nPress any key in the image window to close it.")
                         cv2.waitKey(0)
                         cv2.destroyWindow(window_title)
                     except Exception as display_e:
                         print(f"Error displaying image: {display_e}", file=sys.stderr)
                         cv2.destroyAllWindows()

        else:
            print("Recognition failed. Check server connection and logs.")


    def _display_results(self, frame, faces):
        
        if not faces:
            print(" - No faces data provided to display.")
            
            if self.show_boxes:
                 cv2.imshow('Recognition Result (Press any key to continue)', frame)
            return

        display_frame = frame.copy() 

        print(f"Displaying results for {len(faces)} face(s):")
        for i, face in enumerate(faces):
            name = face.get('name', 'error')
            confidence = float(face.get('confidence', 0.0) or 0.0) * 100
            distance = face.get('distance')
            distance_str = f"{distance:.4f}" if distance is not None else "N/A"
            box_str = str(face.get('box', 'N/A'))

            print(f" - Face {i+1}: Name='{name}', Confidence={confidence:.2f}%, Distance={distance_str}, Box={box_str}")

            if self.show_boxes and name != 'error' and 'box' in face:
                try:
                    x, y, w, h = map(int, face['box'])
                    if name == "unknown": color = (0, 165, 255) 
                    elif name.startswith("error"): color = (0, 0, 255) 
                    else: color = (0, 255, 0) 

                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
                    text = f"{name} ({confidence:.1f}%)"
                    text_y = y - 10 if y - 10 > 10 else y + h + 20
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(display_frame, (x, text_y - th - 5), (x + tw, text_y), color, -1)
                    cv2.putText(display_frame, text, (x, text_y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                except Exception as draw_e:
                     print(f"Warning: Error drawing box/text for face {i+1}: {draw_e}", file=sys.stderr)


        if self.show_boxes:
            try:
                 window_title = 'Recognition Result (Press any key to continue)'
                 cv2.imshow(window_title, display_frame)
                 
                 cv2.waitKey(0)
                 
            except Exception as e:
                 print(f"Error displaying results window: {e}", file=sys.stderr)
                 cv2.destroyAllWindows() 


    def add_face_webcam(self, name):
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.", file=sys.stderr)
            return

        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if frame_width <= 0 or frame_height <= 0:
             print("Error: Invalid frame dimensions from webcam.", file=sys.stderr)
             cap.release()
             return

        
        box_width = int(frame_width * 0.6)
        box_height = int(frame_height * 0.6)
        target_x = (frame_width - box_width) // 2
        target_y = (frame_height - box_height) // 2
        target_box = [target_x, target_y, box_width, box_height] 

        
        
        haar_file = 'haarcascade_frontalface_default.xml'
        face_cascade_path = os.path.join(cv2.data.haarcascades, haar_file)
        if not os.path.exists(face_cascade_path):
            print(f"Error: Haar cascade file not found at {face_cascade_path}", file=sys.stderr)
            print("Ensure OpenCV is installed correctly (e.g., `pip install opencv-python`).", file=sys.stderr)
            cap.release()
            return
        try:
            face_cascade = cv2.CascadeClassifier(face_cascade_path)
            if face_cascade.empty():
                 raise IOError(f"Failed to load Haar cascade file: {face_cascade_path}")
        except Exception as e:
             print(f"Error loading Haar cascade classifier: {e}", file=sys.stderr)
             cap.release()
             return

        recording = False
        countdown_start_sec = 5 
        countdown = 0
        out = None
        video_filepath = None 
        fps = 20.0 

        window_name = f'Face Enrollment for "{name}" (SPACE to Start, Q to Quit)'
        cv2.namedWindow(window_name)

        print(f"\nStarting face enrollment process for '{name}'.")
        print("Position your face clearly within the green box.")
        print("Ensure good lighting and look towards the camera.")
        print("Press SPACE bar when ready to start the 5-second recording.")
        print("Press 'q' to cancel.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.", file=sys.stderr)
                break

            display_frame = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

            in_position = False
            face_large_enough = False

            
            cv2.rectangle(display_frame, (target_box[0], target_box[1]),
                          (target_box[0] + target_box[2], target_box[1] + target_box[3]), (0, 255, 0), 2) 

            for (x, y, w, h) in faces:
                
                face_center_x = x + w // 2
                face_center_y = y + h // 2
                target_cx = target_box[0] + target_box[2] // 2
                target_cy = target_box[1] + target_box[3] // 2

                
                in_horizontal = abs(face_center_x - target_cx) < target_box[2] / 3
                in_vertical = abs(face_center_y - target_cy) < target_box[3] / 3

                
                face_large_enough = h > target_box[3] / 3 

                if in_horizontal and in_vertical:
                    
                    color = (255, 150, 0) if face_large_enough else (0, 0, 255) 
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
                    if face_large_enough:
                        in_position = True
                        break 

            
            y_offset = target_box[1] - 10 
            if recording:
                record_time_left = max(0, countdown)
                cv2.putText(display_frame, f"RECORDING: {record_time_left:.1f}s",
                            (target_box[0], y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) 
                if out and out.isOpened():
                    out.write(frame)
                    countdown -= 1.0 / fps 
                if countdown <= 0:
                    print("\nRecording finished.")
                    break 
            elif in_position:
                cv2.putText(display_frame, "Ready! Press SPACE to record",
                            (target_box[0], y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2) 
            elif faces: 
                 cv2.putText(display_frame, "Adjust position / Get closer",
                             (target_box[0], y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 200), 2) 
            else: 
                cv2.putText(display_frame, "Position face in green box",
                            (target_box[0], y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2) 

            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 32 and not recording and in_position: 
                print("\nStarting recording...")
                
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
                     video_filepath = tmp_file.name
                print(f"Recording to temporary file: {video_filepath}")

                fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
                out = cv2.VideoWriter(video_filepath, fourcc, fps, (frame_width, frame_height))

                if not out or not out.isOpened():
                    print(f"Error: Could not open video writer for {video_filepath}", file=sys.stderr)
                    
                    if video_filepath and os.path.exists(video_filepath):
                        os.remove(video_filepath)
                    break 

                recording = True
                countdown = countdown_start_sec 

            elif key == ord('q'): 
                print("\nEnrollment cancelled by user.")
                recording = False 
                
                if out and out.isOpened():
                    out.release()
                    out = None
                if video_filepath and os.path.exists(video_filepath):
                    try:
                        os.remove(video_filepath)
                        print(f"Deleted incomplete recording: {video_filepath}")
                    except OSError as e:
                         print(f"Warning: Could not delete {video_filepath}: {e}", file=sys.stderr)
                video_filepath = None 
                break 

        
        cap.release()
        if out and out.isOpened():
            out.release()
        cv2.destroyAllWindows()

        
        if video_filepath and os.path.exists(video_filepath) and recording: 
            print(f"Video saved temporarily: {video_filepath}")
            
            self.add_face_file(name, video_filepath)
            
            try:
                os.remove(video_filepath)
                print(f"Removed temporary video file: {video_filepath}")
            except OSError as e:
                print(f"Warning: Could not remove temporary video file {video_filepath}: {e}", file=sys.stderr)
        elif not recording and video_filepath is None: 
             pass 
        else: 
            print("\nEnrollment process did not complete successfully. No video was sent.")


    def add_face_file(self, name, video_path):
        
        if not os.path.exists(video_path):
            print(f"Error: Video file not found at '{video_path}'", file=sys.stderr)
            return

        print(f"\nAdding face for '{name}' using video: {video_path}")
        spinner = Spinner("Uploading video and processing on server...")
        spinner.start()
        response_data = None
        try:
            with open(video_path, 'rb') as f:
                
                mime_type = 'video/mp4' 
                ext = os.path.splitext(video_path)[1].lower()
                if ext == '.avi': mime_type = 'video/x-msvideo'
                elif ext == '.mov': mime_type = 'video/quicktime'
                

                files = {'video': (os.path.basename(video_path), f, mime_type)}
                data = {'name': name}
                response_data = self._make_request('post', 'add_face', files=files, data=data, expect_json=True)

        except IOError as e:
             print(f"Error opening video file {video_path}: {e}", file=sys.stderr)
             spinner.stop()
             return
        finally:
            spinner.stop()

        if response_data:
            status = response_data.get('status', 'No status message received.')
            count = response_data.get('count', 0)
            print(f"\nServer Response:")
            print(f" Status: {status}")
            print(f" Embeddings Added: {count}")
            if count == 0 and "Enrollment failed" not in status:
                print(" -> Note: Zero embeddings added. Check video quality, face visibility, size,")
                print("    and ensure only one dominant face per frame for enrollment.")
        else:
            print("\nFailed to add face. Could not get a valid response from the server.")
            print("Check server connection, logs, and ensure the video format is supported.")


    def list_faces(self):
        
        print("\nRequesting list of registered faces and server info...")
        spinner = Spinner("Fetching data...")
        spinner.start()
        response_data = self._make_request('get', 'list_faces', expect_json=True)
        spinner.stop()

        if response_data:
            
            names = response_data.get('registered_names', [])
            model = response_data.get('active_model', 'N/A')
            collection = response_data.get('collection_name', 'N/A')
            count = response_data.get('total_embeddings', 0)
            detector = response_data.get('detector_backend', 'N/A') 
            min_area = response_data.get('min_face_area_px', 'N/A') 

            print("\n--- Server Status & Registered Faces ---")
            print(f" Server URL: {self.server_url}")
            print(f" Active Model: {model}")
            print(f" Detector Backend: {detector}")
            print(f" Min Face Area: {min_area} px")
            print(f" DB Collection: {collection}")
            print(f" Total Embeddings: {count}")
            print(f"----------------------------------------")
            print(f" Registered Names ({len(names)}):")
            if names:
                
                if len(names) > 20:
                     col_width = max(len(n) for n in names) + 2
                     num_cols = max(1, 80 // col_width)
                     for i in range(0, len(names), num_cols):
                          print(" ".join(f"{n:<{col_width}}" for n in names[i:i+num_cols]))
                else:
                     for name in names:
                         print(f" - {name}")
            else:
                print("   (No faces registered yet)")
            print("----------------------------------------")

            
            self.active_server_model = model
        else:
            print("\nFailed to retrieve list of faces. Check server connection and logs.")


    def remove_face_by_name(self, name):
        
        print(f"\nRequesting removal of ALL entries for name: '{name}'...")
        confirm = input(f"Are you sure you want to delete ALL records for '{name}'? (yes/no): ").lower()
        if confirm != 'yes':
            print("Removal cancelled.")
            return

        spinner = Spinner("Sending removal request...")
        spinner.start()
        data = {'name': name}
        response_data = self._make_request('post', 'remove_face', data=data, expect_json=True)
        spinner.stop()

        if response_data:
            status = response_data.get('status', 'No status message.')
            removed_count = response_data.get('removed_count', 'N/A')
            print(f"\nServer Response:")
            print(f" Status: {status}")
            print(f" Entries Removed: {removed_count}")
        else:
            print(f"\nFailed to remove face '{name}'. Check server connection and logs.")


    def remove_face_by_id(self, ids_to_remove):
        
        if not ids_to_remove:
            print("Error: No IDs provided to remove.", file=sys.stderr)
            return

        print(f"\nRequesting removal of {len(ids_to_remove)} specific ID(s)...")
        print(" IDs:", ", ".join(ids_to_remove))
        confirm = input(f"Are you sure you want to delete these specific entries? (yes/no): ").lower()
        if confirm != 'yes':
            print("Removal cancelled.")
            return

        spinner = Spinner("Sending removal request...")
        spinner.start()
        
        payload = {'ids': ids_to_remove}
        response_data = self._make_request('post', 'remove_id', json=payload, expect_json=True)
        spinner.stop()

        if response_data:
             status = response_data.get('status', 'No status message.')
             requested_ids = response_data.get('requested_ids', []) 
             
             print(f"\nServer Response:")
             print(f" Status: {status}")
             
        else:
            print(f"\nFailed to remove faces by ID. Check server connection and logs.")

    def visualize_embeddings(self, reduction_method='tsne'):
        
        print(f"\nRequesting {reduction_method.upper()} embedding visualization from server...")
        print("(This may take a moment depending on the number of embeddings)")
        spinner = Spinner(f"Generating {reduction_method.upper()} visualization on server...")
        spinner.start()

        
        html_content = self._make_request('get', f'visualize?reduction={reduction_method}', expect_json=False)
        spinner.stop()

        if html_content and isinstance(html_content, str) and html_content.strip().lower().startswith('<!doctype html'):
            try:
                
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.html', encoding='utf-8') as tmp_file:
                    tmp_file.write(html_content)
                    tmp_filepath = tmp_file.name
                print(f"\nVisualization HTML saved temporarily to: {tmp_filepath}")

                
                print("Opening visualization in your web browser...")
                try:
                    webbrowser.open(f'file://{os.path.realpath(tmp_filepath)}') 
                except Exception as wb_err:
                     print(f"Warning: Could not automatically open browser ({wb_err}).", file=sys.stderr)
                     print(f"Please open the file manually: {tmp_filepath}")


                
                input("\nPress Enter after viewing the visualization to close this script (browser window will remain open).")

            except IOError as e:
                 print(f"\nError saving visualization HTML to temporary file: {e}", file=sys.stderr)
            except Exception as e:
                print(f"\nAn unexpected error occurred while handling the visualization: {e}", file=sys.stderr)

        elif html_content:
             
             print("\nReceived unexpected content from server (possibly an error page).", file=sys.stderr)
             
             try:
                 with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='_error.html', encoding='utf-8') as tmp_file:
                     tmp_file.write(str(html_content)) 
                     tmp_filepath = tmp_file.name
                 print(f"Server response saved to: {tmp_filepath}")
                 print("Please inspect this file for error details.")
             except Exception as save_err:
                 print(f"Could not save server response: {save_err}", file=sys.stderr)
                 print("--- Server Response Start ---", file=sys.stderr)
                 print(str(html_content)[:1000] + "...", file=sys.stderr) 
                 print("--- Server Response End ---", file=sys.stderr)

        else:
            
            print("\nFailed to retrieve visualization from server. Check previous errors and server logs.")

    def request_report(self):
        
        print("\nRequesting performance and detail reports from server...")
        spinner = Spinner("Generating reports on server...")
        spinner.start()

        response_data = self._make_request('get', 'report', expect_json=True)
        spinner.stop()

        if not response_data:
            print("\nFailed to retrieve reports. Check server connection and logs.")
            return

        status = response_data.get('status', 'Unknown status.')
        print(f"\nServer Status: {status}")

        if "error" in status.lower() or "failed" in status.lower():
             error_msg = response_data.get('error')
             if error_msg:
                  print(f"Server Error: {error_msg}")
             else:
                  print("An unspecified error occurred on the server during report generation.")
             return 

        graph_b64 = response_data.get('performance_graph_png_base64')
        html_report = response_data.get('detail_report_html')

        saved_files = []

        
        if graph_b64:
            try:
                graph_data = base64.b64decode(graph_b64)
                graph_filename = f"performance_report_{int(time.time())}.png"
                with open(graph_filename, 'wb') as f:
                    f.write(graph_data)
                print(f"Performance graph saved as: {os.path.abspath(graph_filename)}")
                saved_files.append(os.path.abspath(graph_filename))
            except (TypeError, base64.binascii.Error, IOError) as e:
                print(f"Error saving performance graph: {e}", file=sys.stderr)
        else:
            print("Performance graph data was not received from the server.")

        
        if html_report:
            try:
                html_filename = f"recognition_details_{int(time.time())}.html"
                with open(html_filename, 'w', encoding='utf-8') as f:
                    f.write(html_report)
                print(f"Detail report saved as: {os.path.abspath(html_filename)}")
                saved_files.append(os.path.abspath(html_filename))

                
                try:
                     print("Attempting to open detail report in web browser...")
                     webbrowser.open(f'file://{os.path.realpath(html_filename)}')
                except Exception as wb_err:
                     print(f"Warning: Could not automatically open browser ({wb_err}).", file=sys.stderr)
                     print(f"Please open the file manually: {os.path.abspath(html_filename)}")

            except IOError as e:
                print(f"Error saving detail report HTML: {e}", file=sys.stderr)
        else:
            print("Detail report HTML was not received from the server.")

        if not saved_files:
            print("\nNo report files were successfully saved.")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Face Recognition Client - Interact with the Face Server API.',
        formatter_class=argparse.RawTextHelpFormatter 
    )
    parser.add_argument('--server', default=os.environ.get('FACE_SERVER_URL', "http://localhost:5000"),
                        help="URL of the face recognition server (default: http://localhost:5000 or FACE_SERVER_URL env var)")
    parser.add_argument('-v', '--version', action='version', version='%(prog)s 1.0-beta+reports')

    subparsers = parser.add_subparsers(dest='command', required=True,
                                       title='Available Commands',
                                       help='Choose one of the following commands:')

    
    add_parser = subparsers.add_parser('add', help='Add a new face identity via webcam or video file.')
    add_parser.add_argument('--name', required=True, help='Name of the person to add (e.g., "FirstName LastName").')
    add_parser.add_argument('--file', help='Path to a video file (.mp4, .avi, .mov, etc.) for enrollment instead of using the webcam.')

    
    rec_parser = subparsers.add_parser('recognize', help='Recognize faces using the webcam or from an image file.')
    rec_parser.add_argument('--file', help='Path to an image file (.jpg, .png) to recognize faces in.')
    rec_parser.add_argument('--show-boxes', action='store_true', help='Display the image/webcam feed with bounding boxes around detected/recognized faces.')

    
    list_parser = subparsers.add_parser('list', help='List all registered face names and get server status info.')

    
    remove_name_parser = subparsers.add_parser('remove', help='Remove ALL entries associated with a specific face name.')
    remove_name_parser.add_argument('--name', required=True, help='Name of the person whose entries should be removed.')

    
    remove_id_parser = subparsers.add_parser('remove-id', help='Remove specific face entries by their unique database IDs.')
    remove_id_parser.add_argument('--ids', required=True, nargs='+', help='One or more space-separated database IDs to remove.')

    
    viz_parser = subparsers.add_parser('visualize', help='Generate and view an interactive visualization of face embeddings.')
    viz_parser.add_argument('--reduction', default='tsne', choices=['tsne', 'umap'], help='Dimensionality reduction method (default: tsne). UMAP requires `umap-learn` installation.')

    
    report_parser = subparsers.add_parser('report', help='Generate and save performance and detail reports based on logged recognitions.')


    
    try:
        args = parser.parse_args()
    except SystemExit:
         
         sys.exit(0) 

    
    
    show_boxes_flag = getattr(args, 'show_boxes', False) if args.command == 'recognize' else False
    client = FaceClient(args.server, show_boxes=show_boxes_flag)

    
    start_time = time.time()
    try:
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
        elif args.command == 'report':
            client.request_report()
        
        

    except Exception as e:
        print(f"\nAn unexpected error occurred during client operation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc() 
        sys.exit(1)
    finally:
         end_time = time.time()
         print(f"\nCommand '{args.command}' finished in {end_time - start_time:.2f} seconds.")