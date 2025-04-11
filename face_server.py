import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import cv2
import numpy as np
from flask import Flask, request, jsonify
from deepface import DeepFace
import chromadb
from chromadb.config import Settings
import multiprocessing as mp
from multiprocessing import Pool, freeze_support
import atexit

global_pool = None

def init_worker():
    """Preload models in worker processes"""
    try:
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        DeepFace.extract_faces(dummy_frame, detector_backend='mtcnn', enforce_detection=False)
        DeepFace.represent(dummy_frame, model_name="Facenet512", detector_backend='skip')
    except Exception as e:
        logging.error(f"Worker init error: {str(e)}")

def create_process_pool():
    num_workers = max(1, mp.cpu_count() - 1)
    return Pool(processes=num_workers, initializer=init_worker)
app = Flask(__name__)
chroma_client = chromadb.PersistentClient(path="chroma_storage")
collection = chroma_client.get_or_create_collection(
    name="faces",
    metadata={"hnsw:space": "cosine"}
)

@atexit.register
def shutdown_pool():
    if global_pool:
        global_pool.close()
        global_pool.join()
        print("Process pool terminated")

def process_frame(args):
    frame, _ = args
    try:
        faces = DeepFace.extract_faces(frame, detector_backend='mtcnn', enforce_detection=True)
        return [DeepFace.represent(face['face'], model_name="Facenet512", detector_backend='skip')[0]['embedding'] for face in faces]
    except Exception as e:
        logging.warning(f"Frame error: {str(e)}")
        return []

def process_video(video_path, name):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        return {"status": "No frames", "count": 0}

    total = len(frames)
    batch_size = 10
    embeddings = []

    print(f"Processing {total} frames for {name}")
    for i in range(0, total, batch_size):
        batch = frames[i:i+batch_size]
        results = global_pool.map(process_frame, [(f, name) for f in batch])
        batch_embeds = [emb for sublist in results for emb in sublist]
        embeddings.extend(batch_embeds)
        print(f"Processed {min(i+batch_size, total)}/{total}")

    if embeddings:
        collection.add(
            embeddings=embeddings,
            metadatas=[{"name": name}] * len(embeddings),
            ids=[f"{name}_{idx}" for idx in range(len(embeddings))]
        )

    return {"status": f"Added {len(embeddings)}", "count": len(embeddings)}

@app.route('/add_face', methods=['POST'])
def handle_add_face():
    if 'video' not in request.files or 'name' not in request.form:
        return jsonify({"error": "Need video and name"}), 400
    
    try:
        video = request.files['video']
        name = request.form['name']
        temp_path = f"temp_{name}.mp4"
        video.save(temp_path)
        result = process_video(temp_path, name)
        os.remove(temp_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/recognize', methods=['POST'])
def handle_recognize():
    if 'image' not in request.files:
        return jsonify({"error": "Need image"}), 400

    try:
        img_data = request.files['image'].read()
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        results = []

        faces = DeepFace.extract_faces(img, detector_backend='mtcnn', enforce_detection=True)

        for face in faces:
            area = face['facial_area']
            x, y, w, h = area['x'], area['y'], area['w'], area['h']
            
            if w * h < 5000:
                continue

            embedding = DeepFace.represent(face['face'], model_name="Facenet512", detector_backend='skip')[0]['embedding']
            match = collection.query(query_embeddings=[embedding], n_results=1)

            result = {
                "box": [x, y, w, h],
                "distance": match['distances'][0][0],
                "name": match['metadatas'][0][0]['name'] if match['distances'][0][0] < 0.3 else "unknown"
            }
            results.append(result)

        return jsonify({"faces": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/list_faces', methods=['GET'])
def handle_list_faces():
    try:
        faces = collection.get()
        names = list(set(m['name'] for m in faces['metadatas']))
        return jsonify({"faces": names})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/remove_face', methods=['POST'])
def handle_remove_face():
    if 'name' not in request.form:
        return jsonify({"error": "Need name"}), 400
    
    try:
        name = request.form['name']
        collection.delete(where={"name": name})
        return jsonify({"status": f"Removed {name}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    freeze_support()  
    print("Starting server...")
    global_pool = create_process_pool()
    
    try:
        app.run(host='0.0.0.0', port=5000)
    finally:
        shutdown_pool()