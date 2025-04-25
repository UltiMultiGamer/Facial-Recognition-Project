



uuuuuh facial recognition project for uni or sum 

made by meeeee :333333 [@UltiMultiGamer](https://github.com/UltiMultiGamer)  

---

# Setup


1. Make a venv:
   ```bash
   python -m venv venv
   ```
   
2. Boot up the venv:

    ```bash
    .\venv\Scripts\activate
    ```
3. And install dependencies:
   ```bash
   pip install -r requirements.txt
   ```


1. Start the server:  
   ```bash
   python face_server.py
   ```
2. Use `face_client.py` to send the server commands:  
   ```bash
   python face_client.py [command]
   ```

---

# Commands

| Command       | Description                                                               | Example Usage                                                                 |
| :------------ | :------------------------------------------------------------------------ | :---------------------------------------------------------------------------- |
| `add`         | Enroll a new face identity using the webcam or a video file.              | `python face_client.py add --name "Debil"`                                 |
|               |                                                                           | `python face_client.py add --name "Debil" --file "path/to/video.mp4"`    |
| `recognize`   | Recognize faces from the webcam (press Spacebar) or an image file.        | `python face_client.py recognize --show-boxes`                                |
|               |                                                                           | `python face_client.py recognize --file "path/to/image.jpg"`                  |
| `list`        | Retrieve and display registered names and server status information.      | `python face_client.py list`                                                  |
| `remove`      | Remove all embeddings associated with a specific registered name.         | `python face_client.py remove --name "Debil"`                              |
| `remove-id`   | Remove specific embeddings by their unique IDs.                           | `python face_client.py remove-id --ids JaneDoe_167..._0 JohnSmith_167..._5` |
| `visualize`   | Request and open an interactive 2D visualization of embeddings.           | `python face_client.py visualize`                                             |
|               |                                                                           | `python face_client.py visualize --reduction umap`                            |




*   `--server <url>`: Specify the URL of the running face server (default: `http://localhost:5000`).
*   `--name <name>`: The identity name for `add` and `remove` commands.
*   `--file <path>`: Path to a video file (for `add`) or an image file (for `recognize`).
*   `--show-boxes`: (For `recognize`) Display the image/webcam feed with bounding boxes overlaid on detected/recognized faces.
*   `--ids <id1> [<id2>...]`: One or more space-separated embedding IDs (for `remove-id`).
*   `--reduction <tsne|umap>`: The dimensionality reduction algorithm for visualization (default: `tsne`).


# Config

> in face.server.py

DEFAULT_MODEL = "Facenet512" <br> <b>Self-explainatory, use models from list</b>
<br>
DETECTOR_BACKEND = 'mtcnn'<br>
<b>Self-explainatory</b>
<br>
MIN_FACE_AREA_PX = 200*200<br>
<b>Minimum face size area</b>


> in face_client.py


