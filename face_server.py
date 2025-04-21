import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import umap
from sklearn.manifold import TSNE
import plotly.express as px
import pandas as pd
import webbrowser
import tempfile
import time
import atexit
from multiprocessing import Pool, freeze_support
import multiprocessing as mp
from chromadb.config import Settings
import chromadb
from deepface import DeepFace
from flask import Flask, request, jsonify, Response
import numpy as np
import cv2
import logging
import warnings


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


log = logging.getLogger('face_server')
log.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
ch.setFormatter(formatter)
if log.hasHandlers():
    log.handlers.clear()
log.addHandler(ch)
log.propagate = False
logging.getLogger('werkzeug').setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('flask').setLevel(logging.ERROR)
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('chromadb').setLevel(logging.WARNING)
logging.getLogger('deepface').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)


SUPPORTED_MODELS = [
    "VGG-Face", "Facenet", "Facenet512", "OpenFace",
    "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"
]
DEFAULT_MODEL = "Facenet512"
ACTIVE_MODEL = os.environ.get('DEEPFACE_MODEL', DEFAULT_MODEL)
if ACTIVE_MODEL not in SUPPORTED_MODELS:
    log.error(
        f"Error: DEEPFACE_MODEL='{ACTIVE_MODEL}' is not in SUPPORTED_MODELS: {SUPPORTED_MODELS}")
    log.error(f"Defaulting to '{DEFAULT_MODEL}'")
    ACTIVE_MODEL = DEFAULT_MODEL

COLLECTION_NAME = f"faces_{ACTIVE_MODEL.lower().replace('-', '')}"
CHROMA_PATH = "chroma_storage"
DETECTOR_BACKEND = 'mtcnn'
MIN_FACE_SIZE_PX = 15000
NUM_WORKERS = max(1, mp.cpu_count() - 1)


global_pool = None
app = Flask(__name__)


def get_collection_name(model_name):
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Model '{model_name}' is not recognized. Supported: {SUPPORTED_MODELS}")
    return f"faces_{model_name.lower().replace('-', '')}"


def get_chroma_collection(client, model_name):
    collection_name = get_collection_name(model_name)
    try:
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        return collection
    except Exception as e:
        log.error(
            f"Failed to get or create collection '{collection_name}': {e}", exc_info=True)
        return None


def load_chromadb_data_for_viz(collection):
    if not collection:
        log.error("Invalid collection provided to load_chromadb_data_for_viz")
        return None, None, None
    try:
        count = collection.count()
        if count == 0:
            log.warning(
                f"Collection '{collection.name}' is empty. Nothing to visualize.")
            return None, None, None

        log.info(
            f"Fetching {count} items from collection '{collection.name}' for visualization...")

        results = collection.get(include=["embeddings", "metadatas"])

        if not results or not results.get("ids"):
            log.warning(
                "No data retrieved from the collection for visualization (or no IDs found).")
            return None, None, None

        log.info(
            f"Successfully loaded {len(results['ids'])} items for visualization.")
        embeddings = results.get("embeddings")
        metadatas = results.get("metadatas")
        ids = results.get("ids")

        if embeddings is None or metadatas is None or ids is None:
            log.error(
                "Retrieved data missing expected keys (embeddings, metadatas, or ids).")
            return None, None, None

        return embeddings, metadatas, ids

    except Exception as e:
        log.error(
            f"Failed to load data from ChromaDB for visualization: {e}", exc_info=True)
        return None, None, None


def reduce_dimensionality(embeddings, method="tsne", n_components=2):

    if embeddings is None or len(embeddings) == 0:
        log.error("No embeddings provided for dimensionality reduction.")
        return None

    log.info(
        f"Performing dimensionality reduction using {method.upper()} to {n_components}D...")

    embeddings_array = np.array(embeddings)

    if embeddings_array.ndim == 1:
        log.error(
            f"Embeddings array has unexpected shape {embeddings_array.shape}. Expected 2D array.")
        return None
    if embeddings_array.shape[0] == 0:
        log.error("Embeddings array is empty after conversion.")
        return None

    min_samples_tsne = 5
    min_samples_umap = 5

    num_samples = embeddings_array.shape[0]

    if method == "tsne":
        if num_samples < min_samples_tsne:
            log.warning(
                f"Not enough samples ({num_samples}) for t-SNE, need at least {min_samples_tsne}. Skipping reduction.")
            return None

        perplexity = min(30.0, num_samples - 1.0)
        perplexity = max(5.0, perplexity)
        reducer = TSNE(n_components=n_components, random_state=42,
                       perplexity=perplexity, n_iter=300, learning_rate='auto', init='pca')
    elif method == "umap":
        if num_samples < min_samples_umap:
            log.warning(
                f"Not enough samples ({num_samples}) for UMAP, need at least {min_samples_umap}. Skipping reduction.")
            return None

        n_neighbors = min(15, num_samples - 1)
        n_neighbors = max(2, n_neighbors)
        reducer = umap.UMAP(n_components=n_components,
                            n_neighbors=n_neighbors, min_dist=0.1, random_state=42)
    else:
        log.error(
            f"Invalid reduction method '{method}'. Use 'tsne' or 'umap'.")
        return None

    try:
        reduced_embeddings = reducer.fit_transform(embeddings_array)
        log.info("Dimensionality reduction complete.")
        return reduced_embeddings
    except Exception as e:
        log.error(
            f"Error during {method.upper()} reduction: {e}", exc_info=True)
        return None


def init_worker():
    try:
        log.info(
            f"Initializing worker process {os.getpid()} with model '{ACTIVE_MODEL}'...")
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        DeepFace.extract_faces(
            dummy_frame, detector_backend=DETECTOR_BACKEND, enforce_detection=False)
        log.info(f"[{os.getpid()}] Detector '{DETECTOR_BACKEND}' loaded.")
        DeepFace.represent(dummy_frame, model_name=ACTIVE_MODEL,
                           detector_backend='skip', enforce_detection=False)
        log.info(f"[{os.getpid()}] Recognition model '{ACTIVE_MODEL}' loaded.")
    except Exception as e:
        log.error(
            f"Worker init error in process {os.getpid()}: {str(e)}", exc_info=True)


def create_process_pool():
    log.info(f"Creating process pool with {NUM_WORKERS} workers...")
    try:
        pool = Pool(processes=NUM_WORKERS, initializer=init_worker)
        log.info("Process pool created successfully.")
        return pool
    except Exception as e:
        log.error(f"Failed to create process pool: {e}", exc_info=True)
        sys.exit(1)


try:
    log.info(f"Initializing ChromaDB client at path: {CHROMA_PATH}")
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = get_chroma_collection(chroma_client, ACTIVE_MODEL)
    if collection is None:
        log.critical(
            f"Failed to initialize ChromaDB collection for model {ACTIVE_MODEL}. Exiting.")
        sys.exit(1)
    log.info(f"ChromaDB client and collection '{collection.name}' ready.")
    log.info(
        f"Collection '{collection.name}' currently contains {collection.count()} embeddings.")
except Exception as e:
    log.critical(f"Failed to initialize ChromaDB: {e}", exc_info=True)
    sys.exit(1)


@atexit.register
def shutdown_pool():
    global global_pool
    if global_pool:
        log.info("Terminating process pool...")
        global_pool.close()
        global_pool.join()
        log.info("Process pool terminated.")
        global_pool = None


def process_frame_for_embedding(args):
    """
    Extracts faces, checks for multiple large faces, and generates embeddings.
    Returns a list of embeddings if valid, or None if multiple large faces are found.
    """
    frame, frame_idx = args
    embeddings = []
    large_face_count = 0
    valid_faces_info = []

    try:

        faces = DeepFace.extract_faces(frame,
                                       detector_backend=DETECTOR_BACKEND,
                                       enforce_detection=False,
                                       align=True)
        if not faces:

            return []

        for face_info in faces:
            area = face_info['facial_area']
            face_area = area['w'] * area['h']

            if face_area >= MIN_FACE_SIZE_PX:
                large_face_count += 1
                valid_faces_info.append(face_info)

        if large_face_count > 1:
            log.warning(
                f"Frame {frame_idx}: Multiple large faces ({large_face_count}) detected. Rejecting frame for enrollment.")
            return None

        for face_info in valid_faces_info:
            face_img = face_info['face']

            if face_img.size == 0 or face_img.shape[0] < 10 or face_img.shape[1] < 10:
                log.warning(
                    f"Skipping invalid face crop in frame {frame_idx} (size check passed but crop invalid?)")
                continue

            try:
                embedding_obj = DeepFace.represent(face_img,
                                                   model_name=ACTIVE_MODEL,
                                                   enforce_detection=False,
                                                   detector_backend='skip')

                if embedding_obj and 'embedding' in embedding_obj[0]:
                    embeddings.append(embedding_obj[0]['embedding'])
                    log.info(f"Processed embedding {frame_idx}")
                else:
                    log.warning(
                        f"Could not generate embedding for a valid face in frame {frame_idx}")
            except Exception as represent_err:
                log.error(
                    f"Error generating representation for face in frame {frame_idx}: {represent_err}", exc_info=False)

    except ValueError as ve:
        log.warning(f"Value error processing frame {frame_idx}: {str(ve)}")
        return []
    except Exception as e:
        log.error(
            f"Unexpected error processing frame {frame_idx} in worker {os.getpid()}: {str(e)}", exc_info=False)
        return []

    return embeddings


def process_video(video_path, name):

    start_time = time.time()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.error(f"Error opening video file: {video_path}")
        return {"status": "Error opening video file", "count": 0}
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        log.warning(f"Video file '{video_path}' contained no readable frames.")
        return {"status": "No frames found in video", "count": 0}

    total_frames = len(frames)
    log.info(
        f"Processing {total_frames} frames from video for '{name}' using model '{ACTIVE_MODEL}'...")
    frame_args = [(frames[i], i) for i in range(total_frames)]
    all_embeddings = []
    enrollment_failed = False

    try:

        results = global_pool.map(process_frame_for_embedding, frame_args)

        if None in results:
            enrollment_failed = True
            log.error(
                f"Enrollment failed for '{name}': Multiple large faces detected in one or more frames.")
            return {"status": "Enrollment failed: Multiple large faces detected in video", "count": 0}

        all_embeddings = [
            emb for frame_embeds in results if frame_embeds for emb in frame_embeds]

    except Exception as e:
        log.error(
            f"Error during multiprocessing video processing: {e}", exc_info=True)
        return {"status": "Error during processing", "count": 0}

    num_embeddings = len(all_embeddings)
    log.info(
        f"Extracted {num_embeddings} valid embeddings from {total_frames} frames.")

    if num_embeddings > 0:
        try:
            ids = [f"{name}_{int(time.time())}_{idx}" for idx in range(
                num_embeddings)]
            metadatas = [{"name": name, "added_timestamp": time.time()}
                         for _ in range(num_embeddings)]
            collection.add(embeddings=all_embeddings,
                           metadatas=metadatas, ids=ids)
            end_time = time.time()
            log.info(
                f"Successfully added {num_embeddings} embeddings for '{name}' to collection '{collection.name}'. Time taken: {end_time - start_time:.2f}s")
            return {"status": f"Added {num_embeddings} face embeddings for '{name}'", "count": num_embeddings}
        except Exception as e:
            log.error(
                f"Error adding embeddings to ChromaDB for '{name}': {e}", exc_info=True)
            return {"status": "Error storing embeddings", "count": 0}
    else:

        end_time = time.time()
        log.warning(
            f"No valid embeddings were extracted for '{name}'. Time taken: {end_time - start_time:.2f}s. (Check if face size meets threshold: {MIN_FACE_SIZE_PX}px area)")
        return {"status": f"No faces suitable for embedding found for '{name}' (min area: {MIN_FACE_SIZE_PX}px)", "count": 0}


@app.route('/add_face', methods=['POST'])
def handle_add_face():
    if 'video' not in request.files or 'name' not in request.form:
        log.warning("Add face request missing video or name field.")
        return jsonify({"error": "Missing 'video' file or 'name' form field"}), 400
    video_file = request.files['video']
    name = request.form.get('name', '').strip()
    if not name:
        log.warning("Add face: Empty name.")
        return jsonify({"error": "Name cannot be empty"}), 400
    if not video_file or video_file.filename == '':
        log.warning("Add face: No file.")
        return jsonify({"error": "No video file selected"}), 400
    safe_name = "".join(c for c in name if c.isalnum()
                        or c in (' ', '_', '-')).strip()
    if not safe_name:
        log.warning(f"Add face: Invalid name '{name}'.")
        return jsonify({"error": "Invalid name provided"}), 400

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
        temp_path = tmp_file.name
        video_file.save(temp_path)
        log.info(
            f"Received video file for '{safe_name}', saved temporarily to '{temp_path}'")

    result = {}
    try:

        result = process_video(temp_path, safe_name)
        return jsonify(result)
    except Exception as e:
        log.error(
            f"Error in /add_face endpoint for '{safe_name}': {str(e)}", exc_info=True)
        return jsonify({"error": f"An internal server error occurred: {str(e)}"}), 500
    finally:

        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError as e:
                log.error(f"Error removing temporary file {temp_path}: {e}")


@app.route('/recognize', methods=['POST'])
def handle_recognize():
    if 'image' not in request.files:
        return jsonify({"error": "Missing 'image' file"}), 400
    img_file = request.files['image']
    if not img_file or img_file.filename == '':
        return jsonify({"error": "No image file selected"}), 400
    try:
        img_data = img_file.read()
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Could not decode image"}), 400
        log.info(
            f"Received image for recognition ({len(img_data)} bytes). Processing with model '{ACTIVE_MODEL}'.")
        faces = DeepFace.extract_faces(
            img, detector_backend=DETECTOR_BACKEND, enforce_detection=False, align=True)
        if not faces:
            return jsonify({"faces": []})
        recognition_results = []
        for face_info in faces:
            area = face_info['facial_area']
            x, y, w, h = area['x'], area['y'], area['w'], area['h']
            face_img = face_info['face']

            if w * h < MIN_FACE_SIZE_PX:
                continue
            if face_img.size == 0:
                continue
            try:
                embedding_obj = DeepFace.represent(
                    face_img, model_name=ACTIVE_MODEL, enforce_detection=False, detector_backend='skip')
                if not embedding_obj or 'embedding' not in embedding_obj[0]:
                    log.warning(
                        "Could not generate embedding during recognition.")
                    continue
                embedding = embedding_obj[0]['embedding']
                matches = collection.query(query_embeddings=[embedding], n_results=1, include=[
                                           "metadatas", "distances"])
                if matches and matches['ids'] and matches['ids'][0]:
                    match_distance = matches['distances'][0][0]
                    match_metadata = matches['metadatas'][0][0]
                    match_name = match_metadata.get('name', 'unknown')
                    recognition_threshold = 0.40
                    if match_distance < recognition_threshold:
                        identity = match_name
                        confidence = 1 - match_distance
                        log.info(
                            f"Recognized '{identity}' (Dist: {match_distance:.4f})")
                    else:
                        identity = "unknown"
                        confidence = 1 - match_distance
                        log.info(
                            f"Match '{match_name}' too distant (Dist: {match_distance:.4f}). Marked 'unknown'.")
                    result = {"box": [int(x), int(y), int(w), int(h)], "name": identity, "distance": float(
                        match_distance), "confidence": float(confidence)}
                    recognition_results.append(result)
                else:
                    log.info("Face detected, but no matches found in DB.")
                    recognition_results.append({"box": [int(x), int(y), int(w), int(
                        h)], "name": "unknown", "distance": None, "confidence": 0.0})
            except Exception as represent_err:
                log.error(
                    f"Error during representation/query for a face: {represent_err}", exc_info=True)
                recognition_results.append({"box": [int(x), int(y), int(w), int(
                    h)], "name": "error", "distance": None, "confidence": 0.0})
        log.info(
            f"Recognition complete. Found {len(recognition_results)} faces meeting criteria.")
        return jsonify({"faces": recognition_results})
    except ValueError as ve:
        log.warning(f"Value error during recognition: {str(ve)}")
        return jsonify({"error": f"Recognition error: {str(ve)}"}), 400
    except Exception as e:
        log.error(f"Error in /recognize endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route('/list_faces', methods=['GET'])
def handle_list_faces():
    try:
        results = collection.get(include=["metadatas"])
        names = []
        total_embeddings = 0
        if results and results['metadatas']:
            names = sorted(
                list(set(m['name'] for m in results['metadatas'] if 'name' in m)))
            total_embeddings = len(results.get('ids', []))
        log.info(
            f"Listing faces for collection '{collection.name}'. Found {len(names)} unique names, {total_embeddings} total embeddings.")
        return jsonify({"active_model": ACTIVE_MODEL, "collection_name": collection.name, "registered_names": names, "total_embeddings": total_embeddings})
    except Exception as e:
        log.error(f"Error listing faces: {str(e)}", exc_info=True)
        return jsonify({"error": f"An internal server error occurred: {str(e)}"}), 500


@app.route('/remove_face', methods=['POST'])
def handle_remove_face():
    if 'name' not in request.form:
        return jsonify({"error": "Missing 'name' form field"}), 400
    name_to_remove = request.form.get('name', '').strip()
    if not name_to_remove:
        return jsonify({"error": "Name cannot be empty"}), 400
    log.warning(
        f"Received request to remove all entries for name: '{name_to_remove}' from collection '{collection.name}'")
    try:
        results = collection.get(where={"name": name_to_remove}, include=[])
        ids_to_delete = results.get('ids', [])
        if not ids_to_delete:
            log.warning(
                f"No entries found for name '{name_to_remove}' to remove.")
            return jsonify({"status": f"No entries found for name '{name_to_remove}'"})
        collection.delete(ids=ids_to_delete)
        log.info(
            f"Successfully removed {len(ids_to_delete)} entries for name '{name_to_remove}'.")
        return jsonify({"status": f"Removed {len(ids_to_delete)} entries for name '{name_to_remove}'"})
    except Exception as e:
        log.error(
            f"Error removing face data for '{name_to_remove}': {str(e)}", exc_info=True)
        return jsonify({"error": f"An internal server error occurred: {str(e)}"}), 500


@app.route('/remove_id', methods=['POST'])
def handle_remove_id():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    data = request.get_json()
    ids_to_delete = data.get('ids')
    if not ids_to_delete or not isinstance(ids_to_delete, list):
        log.warning("Remove by ID request missing 'ids' list.")
        return jsonify({"error": "Missing 'ids' list in JSON body"}), 400
    ids_to_delete_cleaned = [str(id_val).strip()
                             for id_val in ids_to_delete if str(id_val).strip()]
    if not ids_to_delete_cleaned:
        return jsonify({"error": "No valid IDs provided after cleaning"}), 400
    log.warning(
        f"Received request to remove {len(ids_to_delete_cleaned)} specific IDs from collection '{collection.name}': {ids_to_delete_cleaned}")
    try:
        existing_items = collection.get(ids=ids_to_delete_cleaned, include=[])
        found_ids = existing_items.get('ids', [])
        not_found_ids = list(set(ids_to_delete_cleaned) - set(found_ids))
        if not found_ids:
            log.warning("None of the specified IDs were found.")
            return jsonify({"status": "None of the specified IDs were found"})
        if not_found_ids:
            log.warning(f"IDs not found and skipped: {not_found_ids}")
        collection.delete(ids=found_ids)
        log.info(f"Successfully removed {len(found_ids)} entries by ID.")
        return jsonify({"status": f"Removed {len(found_ids)} entries by ID.", "removed_ids": found_ids, "not_found_ids": not_found_ids})
    except Exception as e:
        log.error(f"Error removing entries by ID: {str(e)}", exc_info=True)
        return jsonify({"error": f"An internal server error occurred: {str(e)}"}), 500


@app.route('/visualize', methods=['GET'])
def handle_visualize():
    log.info(f"Generating visualization for model '{ACTIVE_MODEL}'...")
    reduction_method = request.args.get('reduction', 'tsne').lower()
    if reduction_method not in ['tsne', 'umap']:
        log.warning(
            f"Invalid reduction method: {reduction_method}. Defaulting to tsne.")
        reduction_method = 'tsne'
    try:
        embeddings, metadata, ids = load_chromadb_data_for_viz(collection)
        if embeddings is None:
            return Response("<html><body><h1>Error</h1><p>Could not load data for visualization.</p></body></html>", status=500, mimetype='text/html')
        reduced_embeddings = reduce_dimensionality(
            embeddings, method=reduction_method, n_components=2)
        if reduced_embeddings is None:
            return Response(f"<html><body><h1>Error</h1><p>Could not perform {reduction_method.upper()} reduction.</p></body></html>", status=500, mimetype='text/html')
        log.info("Preparing data for Plotly visualization...")
        names = [meta.get("name", "unknown") if isinstance(
            meta, dict) else "invalid_meta" for meta in metadata]
        df = pd.DataFrame({"x": reduced_embeddings[:, 0], "y": reduced_embeddings[:, 1], "name": names, "id": ids, "metadata": [
                          str(meta) if meta else "None" for meta in metadata]})
        log.info(f"Creating scatter plot for {len(df)} points...")
        fig = px.scatter(df, x="x", y="y", color="name", hover_data=["id", "name", "metadata"], title=f"Face Embeddings Visualization ({ACTIVE_MODEL} Model - {reduction_method.upper()} reduced)", labels={
                         'x': f'{reduction_method.upper()} Dim 1', 'y': f'{reduction_method.upper()} Dim 2'}, template="plotly_white")
        fig.update_layout(legend_title_text='Identity')
        fig.update_traces(marker=dict(size=8, opacity=0.8))
        html_content = fig.to_html(full_html=True, include_plotlyjs='cdn')
        log.info("Visualization HTML generated successfully.")
        return Response(html_content, mimetype='text/html')
    except Exception as e:
        log.error(f"Error generating visualization: {e}", exc_info=True)
        return Response(f"<html><body><h1>Error</h1><p>Internal server error: {e}</p></body></html>", status=500, mimetype='text/html')


@app.route('/status', methods=['GET'])
def handle_status():
    try:
        count = collection.count()
    except Exception as e:
        log.error(f"Could not get collection count for status: {e}")
        count = -1
    return jsonify({"status": "running", "active_model": ACTIVE_MODEL, "collection_name": collection.name, "detector_backend": DETECTOR_BACKEND, "chroma_path": CHROMA_PATH, "collection_embedding_count": count})


if __name__ == '__main__':
    freeze_support()
    log.info("--- Welcome to FRS Ver 1β ! ---")
    log.info(f"Active Model: {ACTIVE_MODEL}")
    log.info(f"Detector Backend: {DETECTOR_BACKEND}")
    log.info(f"Min Face Size (Enroll/Recog): {MIN_FACE_SIZE_PX}px area")
    log.info(f"ChromaDB Collection: {collection.name} at {CHROMA_PATH}")

    global_pool = create_process_pool()
    if not global_pool:
        log.critical("Failed to initialize worker pool. Exiting.")
        sys.exit(1)
    try:
        log.info(f"Starting Flask server on 0.0.0.0:5000")
        app.run(host='0.0.0.0', port=5000, debug=False)
    except Exception as e:
        log.critical(f"Server crashed: {e}", exc_info=True)
    finally:
        shutdown_pool()
        log.info("--- FRS Shutting Down! ---")
