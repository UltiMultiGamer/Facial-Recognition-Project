import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 

import io
import umap 
from sklearn.manifold import TSNE
import plotly.express as px
import pandas as pd
import webbrowser
import tempfile
import time
import atexit
from multiprocessing import Pool, freeze_support, cpu_count
import multiprocessing as mp
from chromadb.config import Settings
import chromadb
from deepface import DeepFace
from flask import Flask, request, jsonify, Response
import numpy as np
import cv2
import logging
import warnings
import shutil 
import uuid 
import base64 
import json 


try:
    import test_utils
except ImportError:
    print("FATAL ERROR: test_utils.py not found. Make sure it's in the same directory.", file=sys.stderr)
    sys.exit(1)



log = logging.getLogger('face_server')
log.setLevel(logging.INFO)

if not log.handlers:
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    ch.setFormatter(formatter)
    log.addHandler(ch)
log.propagate = False 


logging.getLogger('werkzeug').setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('flask').setLevel(logging.ERROR)
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('chromadb').setLevel(logging.WARNING)

logging.getLogger('deepface').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)



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

CHROMA_PATH = "chroma_storage"

os.makedirs(CHROMA_PATH, exist_ok=True)

DETECTOR_BACKEND = 'mtcnn' 
MIN_FACE_AREA_PX = 200*200 
NUM_WORKERS = max(1, cpu_count() - 1) 


RECOGNITION_LOG_IMAGES_DIR = os.path.abspath(test_utils.SERVER_RECOGNITION_LOG_DIR)
REPORT_GENERATION_DIR = os.path.abspath(test_utils.SERVER_REPORT_DIR)
os.makedirs(RECOGNITION_LOG_IMAGES_DIR, exist_ok=True)
os.makedirs(REPORT_GENERATION_DIR, exist_ok=True)
log.info(f"Recognition images will be stored temporarily in: {RECOGNITION_LOG_IMAGES_DIR}")
log.info(f"Generated reports will be stored temporarily in: {REPORT_GENERATION_DIR}")



recognition_log = []



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
        log.info(f"Using ChromaDB collection: '{collection.name}'")
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

        embeddings = results.get("embeddings")
        metadatas = results.get("metadatas")
        ids = results.get("ids")

        
        if embeddings is None or metadatas is None or ids is None:
            log.error("Retrieved data missing expected keys (embeddings, metadatas, or ids).")
            return None, None, None
        if not (len(embeddings) == len(metadatas) == len(ids)):
             log.error("Mismatch in lengths of retrieved embeddings, metadatas, and ids.")
             return None, None, None

        log.info(f"Successfully loaded {len(ids)} items for visualization.")
        return embeddings, metadatas, ids

    except Exception as e:
        log.error(
            f"Failed to load data from ChromaDB for visualization: {e}", exc_info=True)
        return None, None, None

def reduce_dimensionality(embeddings, method="tsne", n_components=2):
    
    if embeddings is None or len(embeddings) == 0:
        log.error("No embeddings provided for dimensionality reduction.")
        return None

    log.info(f"Performing dimensionality reduction using {method.upper()} to {n_components}D...")
    try:
        embeddings_array = np.array(embeddings)
        if embeddings_array.ndim != 2:
            log.error(f"Embeddings array has unexpected shape {embeddings_array.shape}. Expected 2D array.")
            return None
        if embeddings_array.shape[0] == 0:
            log.error("Embeddings array is empty after conversion.")
            return None

        num_samples = embeddings_array.shape[0]
        log.info(f"Reducing {num_samples} embeddings of dimension {embeddings_array.shape[1]}.")

        if method == "tsne":
            min_samples_tsne = max(5, n_components + 1) 
            if num_samples < min_samples_tsne:
                log.warning(f"Not enough samples ({num_samples}) for t-SNE, need at least {min_samples_tsne}. Skipping reduction.")
                return None
            
            perplexity = min(30.0, max(5.0, num_samples / 3.0)) 
            log.info(f"Using t-SNE perplexity: {perplexity:.1f}")
            reducer = TSNE(n_components=n_components, random_state=42, perplexity=perplexity,
                           n_iter=300, learning_rate='auto', init='pca') 

        elif method == "umap":
            min_samples_umap = max(5, n_components + 1)
            if num_samples < min_samples_umap:
                log.warning(f"Not enough samples ({num_samples}) for UMAP, need at least {min_samples_umap}. Skipping reduction.")
                return None
            
            n_neighbors = min(15, max(2, int(num_samples * 0.1))) 
            log.info(f"Using UMAP n_neighbors: {n_neighbors}")
            reducer = UMAP(n_components=n_components, n_neighbors=n_neighbors,
                                min_dist=0.1, random_state=42)
        else:
            log.error(f"Invalid reduction method '{method}'. Use 'tsne' or 'umap'.")
            return None

        reduced_embeddings = reducer.fit_transform(embeddings_array)
        log.info("Dimensionality reduction complete.")
        return reduced_embeddings

    except ImportError as ie:
         log.error(f"Import error during dimensionality reduction: {ie}. Make sure '{method}' library is installed ('pip install {'umap-learn' if method == 'umap' else 'scikit-learn'}').")
         return None
    except Exception as e:
        log.error(f"Error during {method.upper()} reduction: {e}", exc_info=True)
        return None




def init_worker():
    
    pid = os.getpid()
    log.info(f"Initializing worker process {pid} with model '{ACTIVE_MODEL}' and detector '{DETECTOR_BACKEND}'...")
    try:
        
        dummy_frame_detector = np.zeros((100, 100, 3), dtype=np.uint8)
        DeepFace.extract_faces(dummy_frame_detector, detector_backend=DETECTOR_BACKEND, enforce_detection=False)
        log.info(f"[{pid}] Detector '{DETECTOR_BACKEND}' loaded.")

        
        dummy_frame_model = np.zeros((50, 50, 3), dtype=np.uint8) 
        DeepFace.represent(dummy_frame_model, model_name=ACTIVE_MODEL, detector_backend='skip', enforce_detection=False)
        log.info(f"[{pid}] Recognition model '{ACTIVE_MODEL}' loaded.")
        log.info(f"Worker {pid} initialized successfully.")
    except Exception as e:
        log.error(f"Worker init error in process {pid}: {str(e)}", exc_info=True)
        
        

def create_process_pool():
    
    log.info(f"Creating process pool with {NUM_WORKERS} workers...")
    try:
        
        
        pool = Pool(processes=NUM_WORKERS, initializer=init_worker)
        log.info("Process pool created successfully.")
        return pool
    except Exception as e:
        log.critical(f"Failed to create process pool: {e}", exc_info=True)
        sys.exit(1)



try:
    log.info(f"Initializing ChromaDB client at path: {CHROMA_PATH}")
    
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH, settings=Settings(anonymized_telemetry=False))
    collection = get_chroma_collection(chroma_client, ACTIVE_MODEL)
    if collection is None:
        log.critical(f"Failed to initialize ChromaDB collection for model {ACTIVE_MODEL}. Exiting.")
        sys.exit(1)
    log.info(f"ChromaDB client and collection '{collection.name}' ready.")
    log.info(f"Collection '{collection.name}' currently contains {collection.count()} embeddings.")
except Exception as e:
    log.critical(f"Failed to initialize ChromaDB: {e}", exc_info=True)
    sys.exit(1)



def cleanup_resources():
    
    global global_pool
    
    if global_pool:
        log.info("Terminating process pool...")
        try:
            global_pool.close() 
            global_pool.join() 
            log.info("Process pool terminated.")
        except Exception as e:
            log.error(f"Error during pool termination: {e}")
        finally:
            global_pool = None

    
    if os.path.exists(RECOGNITION_LOG_IMAGES_DIR):
        log.info(f"Cleaning up temporary recognition image directory: {RECOGNITION_LOG_IMAGES_DIR}")
        try:
            shutil.rmtree(RECOGNITION_LOG_IMAGES_DIR)
            log.info("Temporary image directory removed.")
        except OSError as e:
            log.error(f"Error removing directory {RECOGNITION_LOG_IMAGES_DIR}: {e}")

    
    if os.path.exists(REPORT_GENERATION_DIR):
        log.info(f"Cleaning up temporary report directory: {REPORT_GENERATION_DIR}")
        try:
            shutil.rmtree(REPORT_GENERATION_DIR)
            log.info("Temporary report directory removed.")
        except OSError as e:
            log.error(f"Error removing directory {REPORT_GENERATION_DIR}: {e}")


atexit.register(cleanup_resources)




def process_frame_for_embedding(args):
    """
    Worker function: Extracts faces, checks criteria, generates embeddings for a single frame.
    Returns a list of valid embeddings found in the frame, or None if enrollment criteria fail (e.g., multiple large faces).
    """
    frame, frame_idx = args
    pid = os.getpid()
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
            
            if 'facial_area' not in face_info or not all(k in face_info['facial_area'] for k in ['w', 'h']):
                 log.warning(f"[{pid}] Frame {frame_idx}: Detected face missing 'facial_area' data. Skipping face.")
                 continue

            area = face_info['facial_area']
            face_area_px = area.get('w', 0) * area.get('h', 0)

            if face_area_px >= MIN_FACE_AREA_PX:
                large_face_count += 1
                valid_faces_info.append(face_info)
            
            


        
        if large_face_count > 1:
            log.warning(f"[{pid}] Frame {frame_idx}: Multiple large faces ({large_face_count}) detected (>= {MIN_FACE_AREA_PX}px). Rejecting frame for enrollment.")
            return None 

        
        for face_info in valid_faces_info:
            face_img = face_info.get('face')

            
            if face_img is None or face_img.size == 0 or face_img.shape[0] < 10 or face_img.shape[1] < 10:
                log.warning(f"[{pid}] Frame {frame_idx}: Skipping invalid face crop detected by DeepFace.")
                continue

            try:
                
                
                embedding_objs = DeepFace.represent(face_img,
                                                    model_name=ACTIVE_MODEL,
                                                    enforce_detection=False,
                                                    detector_backend='skip') 

                
                if embedding_objs and isinstance(embedding_objs, list) and len(embedding_objs) > 0:
                     embedding_data = embedding_objs[0]
                     if 'embedding' in embedding_data and embedding_data['embedding']:
                         embeddings.append(embedding_data['embedding'])
                         
                     else:
                         log.warning(f"[{pid}] Frame {frame_idx}: represent() call succeeded but no 'embedding' found in result.")
                else:
                    log.warning(f"[{pid}] Frame {frame_idx}: represent() call did not return expected embedding data.")

            except ValueError as ve:
                 
                 log.warning(f"[{pid}] Frame {frame_idx}: ValueError during face representation: {ve}. Check detector_backend ('skip' expected).")
                 continue 
            except Exception as represent_err:
                log.error(f"[{pid}] Frame {frame_idx}: Error generating representation for a face: {represent_err}", exc_info=False)
                continue 

    except ValueError as ve:
        
        
        log.warning(f"[{pid}] ValueError processing frame {frame_idx}: {str(ve)}")
        return [] 
    except Exception as e:
        log.error(f"[{pid}] Unexpected error processing frame {frame_idx}: {str(e)}", exc_info=False)
        return [] 

    
    return embeddings




def process_video(video_path, name):
    
    global global_pool, collection 

    start_time_total = time.time()
    log.info(f"Starting video processing for '{name}' from path: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.error(f"Error opening video file: {video_path}")
        return {"status": "Error opening video file", "count": 0}

    frames = []
    frame_count = 0
    max_frames_to_read = 300 
    while frame_count < max_frames_to_read:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        frame_count += 1
    cap.release()

    if not frames:
        log.warning(f"Video file '{video_path}' contained no readable frames (or limit reached).")
        return {"status": "No frames found in video", "count": 0}

    total_frames_read = len(frames)
    log.info(f"Read {total_frames_read} frames. Preparing for parallel processing...")
    log.info(f"Using Model: '{ACTIVE_MODEL}', Detector: '{DETECTOR_BACKEND}', Min Face Area: {MIN_FACE_AREA_PX}px")

    
    frame_args = [(frames[i], i) for i in range(total_frames_read)]
    all_embeddings = []
    enrollment_failed_multiple_faces = False

    if not global_pool:
         log.error("Process pool is not available. Cannot process video.")
         return {"status": "Internal server error: Process pool not initialized", "count": 0}

    try:
        log.info(f"Distributing {len(frame_args)} frame tasks to {NUM_WORKERS} workers...")
        start_time_pool = time.time()

        
        results = global_pool.map(process_frame_for_embedding, frame_args)

        pool_duration = time.time() - start_time_pool
        log.info(f"Worker pool processing finished in {pool_duration:.2f}s.")

        
        valid_frame_results = 0
        for result in results:
            if result is None:
                enrollment_failed_multiple_faces = True
                break 
            elif isinstance(result, list):
                all_embeddings.extend(result)
                if result: 
                    valid_frame_results += 1
            

        if enrollment_failed_multiple_faces:
            log.error(f"Enrollment FAILED for '{name}': Multiple large faces detected in one or more frames.")
            return {"status": "Enrollment failed: Multiple large faces detected in video", "count": 0}

    except Exception as e:
        log.error(f"Error during multiprocessing video processing for '{name}': {e}", exc_info=True)
        return {"status": "Error during processing", "count": 0}

    num_embeddings = len(all_embeddings)
    log.info(f"Extracted {num_embeddings} valid embeddings from {valid_frame_results}/{total_frames_read} processed frames.")

    
    if num_embeddings > 0:
        try:
            
            timestamp_sec = int(time.time())
            ids = [f"{name}_{timestamp_sec}_{idx}" for idx in range(num_embeddings)]
            metadatas = [{"name": name, "added_timestamp": time.time()} for _ in range(num_embeddings)]
            log.info(f"Adding {num_embeddings} embeddings for '{name}' to collection '{collection.name}'...")
            start_time_db = time.time()
            collection.add(embeddings=all_embeddings, metadatas=metadatas, ids=ids)
            db_duration = time.time() - start_time_db
            total_duration = time.time() - start_time_total
            log.info(f"Successfully added embeddings to ChromaDB in {db_duration:.2f}s.")
            log.info(f"Total processing time for '{name}': {total_duration:.2f}s")
            return {"status": f"Added {num_embeddings} face embeddings for '{name}'", "count": num_embeddings}
        except Exception as e:
            log.error(f"Error adding embeddings to ChromaDB for '{name}': {e}", exc_info=True)
            return {"status": "Error storing embeddings", "count": 0}
    else:
        
        total_duration = time.time() - start_time_total
        log.warning(f"No valid embeddings extracted for '{name}'. Check video quality, face size/visibility, and logs.")
        log.info(f"Total processing time for '{name}': {total_duration:.2f}s")
        status_msg = (f"No faces suitable for embedding found for '{name}'. "
                      f"Ensure clear faces meeting minimum size ({MIN_FACE_AREA_PX}px area) "
                      f"and only one large face per frame.")
        return {"status": status_msg, "count": 0}




@app.route('/add_face', methods=['POST'])
def handle_add_face():
    
    if 'video' not in request.files or 'name' not in request.form:
        log.warning("Add face request missing 'video' file or 'name' form field.")
        return jsonify({"error": "Missing 'video' file or 'name' form field"}), 400

    video_file = request.files['video']
    name = request.form.get('name', '').strip()

    if not name:
        log.warning("Add face request received with empty name.")
        return jsonify({"error": "Name cannot be empty"}), 400
    if not video_file or video_file.filename == '':
        log.warning("Add face request received with no file selected.")
        return jsonify({"error": "No video file selected"}), 400

    
    safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '_', '-')).strip()
    if not safe_name:
        log.warning(f"Add face request provided invalid name after sanitization: '{name}'.")
        return jsonify({"error": "Invalid characters in name"}), 400

    
    temp_path = None
    try:
        
        _, file_extension = os.path.splitext(video_file.filename)
        with tempfile.NamedTemporaryFile(suffix=file_extension or ".mp4", delete=False) as tmp_file:
            temp_path = tmp_file.name
            video_file.save(temp_path)
            log.info(f"Received video for '{safe_name}', saved temporarily to '{temp_path}'")

        
        result = process_video(temp_path, safe_name)
        return jsonify(result)

    except Exception as e:
        log.error(f"Error in /add_face endpoint for '{safe_name}': {str(e)}", exc_info=True)
        return jsonify({"error": f"An internal server error occurred processing the video."}), 500
    finally:
        
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                log.info(f"Removed temporary video file: {temp_path}")
            except OSError as e:
                log.error(f"Error removing temporary file {temp_path}: {e}")


@app.route('/recognize', methods=['POST'])
def handle_recognize():
    start_time = time.time()
    if 'image' not in request.files:
        return jsonify({"error": "Missing 'image' file"}), 400

    img_file = request.files['image']
    if not img_file or img_file.filename == '':
        return jsonify({"error": "No image file selected"}), 400

    temp_image_path = None 
    try:
        img_data = img_file.read()
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            log.warning("Could not decode received image data.")
            return jsonify({"error": "Could not decode image"}), 400

        # Get original filename and add timestamp
        original_filename = img_file.filename
        timestamp = int(time.time())
        safe_filename = f"{os.path.splitext(original_filename)[0]}_{timestamp}{os.path.splitext(original_filename)[1]}"
        
        img_filename = f"rec_{safe_filename}"
        temp_image_path = os.path.join(RECOGNITION_LOG_IMAGES_DIR, img_filename)
        cv2.imwrite(temp_image_path, img)
        log.info(f"Received image ({len(img_data)} bytes), saved to {temp_image_path}. Processing...")

        # Process the image...
        faces = DeepFace.extract_faces(
            img,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,
            align=True 
        )

        if not faces:
            log.info("No faces detected in the image.")
            
            rec_time = time.time() - start_time
            recognition_log.append({
                'image_path': temp_image_path,
                'original_filename': original_filename,  # Store original filename
                'results': [],
                'time': rec_time,
                'timestamp': start_time
            })
            return jsonify({"faces": []})

        # Process faces and create recognition results...
        recognition_results = []
        log.info(f"Detected {len(faces)} faces. Checking size and attempting recognition...")
        for face_info in faces:
            area = face_info.get('facial_area')
            face_img = face_info.get('face')

            if not area or not all(k in area for k in ['x', 'y', 'w', 'h']):
                 log.warning("Skipping face with missing 'facial_area' data.")
                 continue
            x, y, w, h = area['x'], area['y'], area['w'], area['h']

            
            if w * h < MIN_FACE_AREA_PX:
                log.info(f"Skipping face below min area: {w*h}px < {MIN_FACE_AREA_PX}px at [{x},{y}]")
                continue

            if face_img is None or face_img.size == 0:
                log.warning(f"Skipping face with invalid crop at [{x},{y}]")
                continue

            face_result = {"box": [int(x), int(y), int(w), int(h)], "name": "error", "distance": None, "confidence": 0.0}

            try:
                
                embedding_objs = DeepFace.represent(
                    face_img,
                    model_name=ACTIVE_MODEL,
                    enforce_detection=False, 
                    detector_backend='skip'  
                )

                if not embedding_objs or not isinstance(embedding_objs, list) or len(embedding_objs) == 0 or 'embedding' not in embedding_objs[0] or not embedding_objs[0]['embedding']:
                    log.warning(f"Could not generate embedding for face at [{x},{y}].")
                    face_result["name"] = "error_embedding"
                    recognition_results.append(face_result)
                    continue

                embedding = embedding_objs[0]['embedding']

                
                try:
                    matches = collection.query(
                        query_embeddings=[embedding],
                        n_results=1, 
                        include=["metadatas", "distances"]
                    )
                except Exception as db_err:
                    log.error(f"Database query failed for face at [{x},{y}]: {db_err}")
                    face_result["name"] = "error_db_query"
                    recognition_results.append(face_result)
                    continue

                
                if matches and matches.get('ids') and matches['ids'][0]:
                    
                    
                    
                    
                    
                    
                    recognition_threshold = 0.60 if ACTIVE_MODEL in ["Facenet512", "ArcFace", "SFace"] else 0.40
                    
                    max_distance = 2.0 

                    match_distance = matches['distances'][0][0]
                    match_metadata = matches['metadatas'][0][0]
                    match_name = match_metadata.get('name', 'unknown_db_entry')

                    if match_distance < recognition_threshold:
                        identity = match_name
                        
                        
                        
                        confidence = max(0.0, min(1.0, 1.0 - (match_distance / recognition_threshold))) 
                        log.info(f"Recognized '{identity}' at [{x},{y}] (Dist: {match_distance:.4f} < {recognition_threshold}, Conf: {confidence:.2f})")
                    else:
                        identity = "unknown"
                        
                        confidence = max(0.0, min(1.0, 1.0 - (match_distance / max_distance))) 
                        log.info(f"Match '{match_name}' too distant (Dist: {match_distance:.4f} >= {recognition_threshold}). Marked 'unknown' at [{x},{y}].")

                    face_result["name"] = identity
                    face_result["distance"] = float(match_distance)
                    face_result["confidence"] = float(confidence)

                else:
                    
                    log.info(f"Face detected at [{x},{y}], but no matches found in DB.")
                    face_result["name"] = "unknown"
                    face_result["distance"] = None 
                    face_result["confidence"] = 0.0 

                recognition_results.append(face_result)

            except ValueError as ve:
                log.warning(f"ValueError during representation/query for face at [{x},{y}]: {str(ve)}")
                face_result["name"] = "error_represent_val"
                recognition_results.append(face_result)
            except Exception as e:
                log.error(f"Unexpected error processing face at [{x},{y}]: {e}", exc_info=True)
                face_result["name"] = "error_unknown"
                recognition_results.append(face_result)

        
        rec_time = time.time() - start_time
        recognition_log.append({
            'image_path': temp_image_path,
            'original_filename': original_filename,  # Store original filename
            'results': recognition_results, 
            'time': rec_time,
            'timestamp': start_time
        })

        log.info(f"Recognition complete for {img_filename}. Found {len(recognition_results)} faces meeting criteria. Time: {rec_time:.3f}s")
        return jsonify({"faces": recognition_results})

    except ValueError as ve:
        
        log.warning(f"Value error during recognition setup: {str(ve)}")
        
        if temp_image_path and os.path.exists(temp_image_path): os.remove(temp_image_path)
        return jsonify({"error": f"Recognition error: {str(ve)}"}), 400
    except Exception as e:
        log.error(f"Critical error in /recognize endpoint: {str(e)}", exc_info=True)
        
        if temp_image_path and os.path.exists(temp_image_path): os.remove(temp_image_path)
        return jsonify({"error": f"Internal server error during recognition."}), 500


@app.route('/list_faces', methods=['GET'])
def handle_list_faces():
    
    try:
        
        results = collection.get(include=["metadatas"]) 

        names = set()
        total_embeddings = 0
        if results and results.get('metadatas'):
            metadatas = results['metadatas']
            total_embeddings = len(results.get('ids', [])) 
            for meta in metadatas:
                if isinstance(meta, dict) and 'name' in meta:
                    names.add(meta['name'])

        sorted_names = sorted(list(names))
        log.info(f"Listing faces for collection '{collection.name}'. Found {len(sorted_names)} unique names, {total_embeddings} total embeddings.")
        return jsonify({
            "active_model": ACTIVE_MODEL,
            "collection_name": collection.name,
            "registered_names": sorted_names,
            "total_embeddings": total_embeddings
        })
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

    log.warning(f"Received request to remove ALL entries for name: '{name_to_remove}' from collection '{collection.name}'")

    try:
        
        results = collection.get(where={"name": name_to_remove}, include=[]) 
        ids_to_delete = results.get('ids', [])

        if not ids_to_delete:
            log.warning(f"No entries found for name '{name_to_remove}' to remove.")
            return jsonify({"status": f"No entries found for name '{name_to_remove}'", "removed_count": 0})

        log.info(f"Found {len(ids_to_delete)} entries for name '{name_to_remove}'. Proceeding with deletion...")
        collection.delete(ids=ids_to_delete)
        log.info(f"Successfully removed {len(ids_to_delete)} entries for name '{name_to_remove}'.")
        
        
        
        
        
        

        return jsonify({"status": f"Removed {len(ids_to_delete)} entries for name '{name_to_remove}'", "removed_count": len(ids_to_delete)})

    except Exception as e:
        log.error(f"Error removing face data for '{name_to_remove}': {str(e)}", exc_info=True)
        return jsonify({"error": f"An internal server error occurred: {str(e)}"}), 500


@app.route('/remove_id', methods=['POST'])
def handle_remove_id():
    
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    ids_to_delete = data.get('ids')

    if not ids_to_delete or not isinstance(ids_to_delete, list):
        log.warning("Remove by ID request missing 'ids' list in JSON body.")
        return jsonify({"error": "Missing 'ids' list in JSON body"}), 400

    
    ids_to_delete_cleaned = [str(id_val).strip() for id_val in ids_to_delete if str(id_val).strip()]
    if not ids_to_delete_cleaned:
        return jsonify({"error": "No valid IDs provided after cleaning"}), 400

    log.warning(f"Received request to remove {len(ids_to_delete_cleaned)} specific IDs from collection '{collection.name}'") 

    try:
        
        
        
        
        log.info(f"Attempting deletion of {len(ids_to_delete_cleaned)} IDs...")
        collection.delete(ids=ids_to_delete_cleaned)

        
        
        
        log.info(f"Deletion request sent for {len(ids_to_delete_cleaned)} IDs.")
        
        return jsonify({"status": f"Deletion request processed for {len(ids_to_delete_cleaned)} IDs.", "requested_ids": ids_to_delete_cleaned})

    except Exception as e:
        log.error(f"Error removing entries by ID: {str(e)}", exc_info=True)
        return jsonify({"error": f"An internal server error occurred: {str(e)}"}), 500


@app.route('/visualize', methods=['GET'])
def handle_visualize():
    
    log.info(f"Generating visualization for model '{ACTIVE_MODEL}'...")
    reduction_method = request.args.get('reduction', 'tsne').lower()
    if reduction_method not in ['tsne', 'umap']:
        log.warning(f"Invalid reduction method: {reduction_method}. Defaulting to tsne.")
        reduction_method = 'tsne'

    try:
        embeddings, metadata, ids = load_chromadb_data_for_viz(collection)

        if embeddings is None or not ids:
            log.warning("No data loaded for visualization.")
            
            html_no_data = """
            <html><head><title>Visualization Error</title></head>
            <body><h1>Visualization Error</h1>
            <p>Could not load data for visualization. The collection might be empty or an error occurred.</p>
            </body></html>"""
            return Response(html_no_data, status=404, mimetype='text/html')

        reduced_embeddings = reduce_dimensionality(embeddings, method=reduction_method, n_components=2)

        if reduced_embeddings is None:
            log.error(f"Could not perform {reduction_method.upper()} reduction.")
            html_error = f"""
            <html><head><title>Visualization Error</title></head>
            <body><h1>Visualization Error</h1>
            <p>Could not perform {reduction_method.upper()} dimensionality reduction. Check server logs for details. Ensure necessary libraries (scikit-learn/umap-learn) are installed.</p>
            </body></html>"""
            return Response(html_error, status=500, mimetype='text/html')

        log.info("Preparing data for Plotly visualization...")
        
        names = [meta.get("name", "unknown") if isinstance(meta, dict) else "invalid_meta" for meta in metadata]
        
        hover_texts = [json.dumps(meta, indent=2) if isinstance(meta, dict) else str(meta) for meta in metadata]

        df = pd.DataFrame({
            "x": reduced_embeddings[:, 0],
            "y": reduced_embeddings[:, 1],
            "name": names,
            "id": ids,
            "metadata_hover": hover_texts 
        })

        log.info(f"Creating scatter plot for {len(df)} points...")
        fig = px.scatter(df, x="x", y="y", color="name",
                         hover_data={"id": True, "name": True, "metadata_hover": True, "x": False, "y": False}, 
                         title=f"Face Embeddings Visualization ({ACTIVE_MODEL} Model - {reduction_method.upper()} reduced)",
                         labels={'x': f'{reduction_method.upper()} Dim 1', 'y': f'{reduction_method.upper()} Dim 2', 'metadata_hover': 'Metadata'},
                         template="plotly_white") 

        fig.update_layout(legend_title_text='Identity')
        fig.update_traces(marker=dict(size=8, opacity=0.8, line=dict(width=0.5, color='DarkSlateGrey'))) 

        
        html_content = fig.to_html(full_html=True, include_plotlyjs='cdn')

        log.info("Visualization HTML generated successfully.")
        
        return Response(html_content, mimetype='text/html')

    except Exception as e:
        log.error(f"Error generating visualization: {e}", exc_info=True)
        html_fatal = f"""
        <html><head><title>Visualization Error</title></head>
        <body><h1>Internal Server Error</h1>
        <p>An unexpected error occurred while generating the visualization: {e}</p>
        <p>Please check the server logs.</p>
        </body></html>"""
        return Response(html_fatal, status=500, mimetype='text/html')


@app.route('/report', methods=['GET'])
def handle_report():
    log.info("Received request to generate reports...")

    if not recognition_log:
        log.warning("No recognition events logged. Cannot generate reports.")
        return jsonify({"error": "No recognition events have been processed yet."}), 404

    # Skip the first result (warmup)
    perf_data = []
    event_counter = 0
    for log_entry in recognition_log[1:]:
        event_id = f"rec_{event_counter}_{int(log_entry['timestamp'])}"
        event_counter += 1
        # Use original filename if available
        img_basename = log_entry.get('original_filename', os.path.basename(log_entry.get('image_path', 'unknown_image')))

        if not log_entry.get('results'): 
             perf_data.append({
                'event_id': f"{event_id}_noface",
                'image': img_basename,
                'name': 'N/A',
                'confidence': None,
                'recognition_time': log_entry.get('time', 0.0)
             })
        else:
             for i, face_res in enumerate(log_entry['results']):
                 perf_data.append({
                    'event_id': f"{event_id}_face{i}",
                    'image': img_basename,
                    'name': face_res.get('name', 'error'),
                    'confidence': face_res.get('confidence'), 
                    'recognition_time': log_entry.get('time', 0.0) 
                 })

    if not perf_data:
         log.warning("Could not extract any performance data from logs.")
         
         return jsonify({"error": "Failed to extract performance data from logs."}), 500

    perf_df = pd.DataFrame(perf_data)
    
    perf_df.reset_index(inplace=True) 

    
    perf_graph_path = os.path.join(REPORT_GENERATION_DIR, f"performance_report_{int(time.time())}.png")
    try:
        log.info("Generating performance visualization graph...")
        test_utils.create_performance_visualization(perf_df, perf_graph_path)
        log.info(f"Performance graph generated: {perf_graph_path}")
        
        perf_graph_base64 = test_utils.image_to_base64(perf_graph_path)
        if not perf_graph_base64:
            raise ValueError("Failed to encode performance graph to Base64.")
    except Exception as e:
        log.error(f"Failed to generate or encode performance graph: {e}", exc_info=True)
        return jsonify({"error": "Failed to generate performance graph."}), 500

    html_content = ["<html><head><title>Отчет о распознавании лиц</title>",
                    "<style>",
                    "body { font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }",
                    "h1 { color: #2c3e50; text-align: center; margin-bottom: 30px; }",
                    "h2 { color: #34495e; margin-top: 30px; padding-bottom: 10px; border-bottom: 2px solid #3498db; }",
                    "h3 { color: #2c3e50; margin-top: 20px; }",
                    ".event { background-color: white; border-radius: 8px; padding: 20px; margin-bottom: 30px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }",
                    ".event-meta { font-size: 0.9em; color: #7f8c8d; margin: 10px 0; }",
                    ".images { display: flex; gap: 20px; align-items: flex-start; flex-wrap: wrap; margin: 20px 0; }",
                    ".images img { max-width: 45%; height: auto; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); transition: transform 0.3s ease; }",
                    ".images img:hover { transform: scale(1.02); }",
                    ".results { background-color: #f8f9fa; border-radius: 4px; padding: 15px; margin: 15px 0; overflow-x: auto; }",
                    ".results pre { margin: 0; font-family: 'Consolas', monospace; font-size: 0.9em; }",
                    ".header-info { background-color: white; padding: 20px; border-radius: 8px; margin-bottom: 30px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }",
                    ".header-info p { margin: 5px 0; color: #34495e; }",
                    ".error-message { color: #e74c3c; background-color: #fde8e8; padding: 10px; border-radius: 4px; margin: 10px 0; }",
                    ".success-message { color: #27ae60; background-color: #e8f5e9; padding: 10px; border-radius: 4px; margin: 10px 0; }",
                    ".image-container { display: flex; flex-direction: column; align-items: center; }",
                    ".image-container div { margin-top: 10px; font-size: 0.9em; color: #7f8c8d; }",
                    "</style></head><body>"]
    
    html_content.append(f"<h1>Отчет о распознавании лиц</h1>")
    html_content.append("<div class='header-info'>")
    html_content.append(f"<p><strong>Дата генерации:</strong> {time.strftime('%Y-%m-%d %H:%M:%S')}</p>")
    html_content.append(f"<p><strong>Используемая модель:</strong> {ACTIVE_MODEL}</p>")
    html_content.append(f"<p><strong>Детектор:</strong> {DETECTOR_BACKEND}</p>")
    html_content.append(f"<p><strong>Всего событий распознавания:</strong> {len(recognition_log)-1}</p>")
    html_content.append("</div>")

    log.info("Generating detailed HTML report...")
    report_generation_success = True
    
    for i, log_entry in enumerate(recognition_log[1:], start=1):
        event_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(log_entry['timestamp']))
        image_path = log_entry['image_path']
        results_json = log_entry['results']
        processing_time = log_entry['time']
        original_filename = log_entry.get('original_filename', os.path.basename(image_path))

        if not os.path.exists(image_path):
            log.warning(f"Image file not found for report entry {i}: {image_path}. Skipping visualization.")
            html_content.append(f"<div class='event'><h2>Событие {i} ({event_time}) - Ошибка</h2>")
            html_content.append(f"<p class='event-meta'>Изображение: {original_filename} (Файл не найден!)</p>")
            html_content.append(f"<p class='event-meta'>Время обработки: {processing_time:.3f}с</p>")
            html_content.append("</div>")
            continue 

        viz_img_filename = f"viz_{os.path.basename(image_path)}"
        viz_img_path = os.path.join(REPORT_GENERATION_DIR, viz_img_filename)
        viz_success = test_utils.visualize_recognition_process(image_path, results_json, viz_img_path)

        html_content.append(f"<div class='event'><h2>Событие {i} ({event_time})</h2>")
        html_content.append(f"<p class='event-meta'>Изображение: {original_filename}</p>")
        html_content.append(f"<p class='event-meta'>Время обработки: {processing_time:.3f}с</p>")

        html_content.append("<div class='images'>")
        
        original_b64 = test_utils.image_to_base64(image_path)
        if original_b64:
            html_content.append(f"<div class='image-container'>")
            html_content.append(f"<img src='data:image/jpeg;base64,{original_b64}' alt='Original Image {i}'>")
            html_content.append("<div>Оригинальное изображение</div>")
            html_content.append("</div>")
        else:
            html_content.append("<div class='error-message'>Ошибка: не удалось закодировать оригинальное изображение</div>")

        if viz_success and os.path.exists(viz_img_path):
            viz_b64 = test_utils.image_to_base64(viz_img_path)
            if viz_b64:
                html_content.append(f"<div class='image-container'>")
                html_content.append(f"<img src='data:image/png;base64,{viz_b64}' alt='Processed Image {i}'>")
                html_content.append("<div>Результат распознавания</div>")
                html_content.append("</div>")
            else:
                html_content.append("<div class='error-message'>Ошибка: не удалось закодировать обработанное изображение</div>")
            
            os.remove(viz_img_path)
        elif viz_success is None: 
            html_content.append("<div class='error-message'>Ошибка: не удалось визуализировать результаты</div>")
            report_generation_success = False
        else: 
            html_content.append("<div class='error-message'>Ошибка: файл не найден после генерации</div>")

        html_content.append("</div>")

        html_content.append("<h3>Результаты распознавания (JSON):</h3>")
        html_content.append(f"<div class='results'><pre>{json.dumps(results_json, indent=2)}</pre></div>")

        html_content.append("</div>")

    html_content.append("</body></html>")

    
    try:
        with open(html_report_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(html_content))
        log.info(f"HTML report generated: {html_report_path}")

        with open(html_report_path, 'r', encoding='utf-8') as f:
            html_report_content = f.read()

    except Exception as e:
        log.error(f"Failed to generate or read HTML report: {e}", exc_info=True)
        report_generation_success = False
        html_report_content = "<html><body><h1>Error generating HTML report.</h1></body></html>" 

    log.info("Sending report data to client.")
    return jsonify({
        "status": "Reports generated" + (" with errors" if not report_generation_success else ""),
        "performance_graph_png_base64": perf_graph_base64,
        "detail_report_html": html_report_content
    })


def warmup_server(flask_app):
    
    log.info("--- Warming up /recognize endpoint using warmup.jpg ---")
    warmup_filename = "warmup.jpg"

    
    if not os.path.exists(warmup_filename):
        log.error(f"Warmup failed: File '{warmup_filename}' not found in the project directory.")
        log.error("Please ensure the image exists where face_server.py is run.")
        log.info("--- Warmup skipped ---")
        return

    try:
        
        test_client = flask_app.test_client()

        
        with open(warmup_filename, 'rb') as f:
            
            response = test_client.post(
                '/recognize',
                data={'image': (f, warmup_filename)}, 
                content_type='multipart/form-data' 
            )

        
        if response.status_code == 200:
            
            response_summary = response.get_data(as_text=True)
            if len(response_summary) > 150:
                 response_summary = response_summary[:150] + "..."
            log.info(f"Warmup request successful (Status: {response.status_code}). Response: {response_summary}")
        else:
            log.error(f"Warmup request failed (Status: {response.status_code}). Response: {response.get_data(as_text=True)}")

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    except Exception as e:
        log.error(f"Exception during server warmup with {warmup_filename}: {e}", exc_info=True)

    log.info("--- Warmup attempt finished ---")
@app.route('/status', methods=['GET'])
def handle_status():
    
    collection_count = -1
    try:
        collection_count = collection.count()
    except Exception as e:
        log.error(f"Could not get collection count for status: {e}")

    status_info = {
        "status": "running",
        "active_model": ACTIVE_MODEL,
        "collection_name": collection.name,
        "detector_backend": DETECTOR_BACKEND,
        "min_face_area_px": MIN_FACE_AREA_PX,
        "chroma_path": os.path.abspath(CHROMA_PATH),
        "collection_embedding_count": collection_count,
        "worker_processes": NUM_WORKERS,
        "recognition_log_count": len(recognition_log) 
    }
    return jsonify(status_info)



if __name__ == '__main__':
    freeze_support() 
    log.info("--- FRS Server Starting ---")
    log.info(f"Version: 1.0-br")
    log.info(f"Active Model: {ACTIVE_MODEL}")
    log.info(f"Detector Backend: {DETECTOR_BACKEND}")
    log.info(f"Min Face Area (Enroll/Recog): {MIN_FACE_AREA_PX}px")
    log.info(f"ChromaDB Collection: {collection.name} at {CHROMA_PATH}")
    log.info(f"Worker Processes: {NUM_WORKERS}")
    log.info(f"Temporary Recognition Image Dir: {RECOGNITION_LOG_IMAGES_DIR}")
    log.info(f"Temporary Report Generation Dir: {REPORT_GENERATION_DIR}")


    
    global_pool = create_process_pool()
    if not global_pool:
        log.critical("Failed to initialize worker pool. Exiting.")
        sys.exit(1)
    warmup_server(app)
    try:
        log.info(f"Starting Flask server on 0.0.0.0:5000 (Press CTRL+C to stop)")
        
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        log.info("CTRL+C received. Shutting down server...")
    except Exception as e:
        log.critical(f"Server crashed unexpectedly: {e}", exc_info=True)
    finally:
        
        log.info("--- FRS Server Shutting Down ---")