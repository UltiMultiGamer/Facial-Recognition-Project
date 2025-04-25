import os
import time
import cv2
import numpy as np
import matplotlib
try:
    matplotlib.use('Agg')
    print("Matplotlib backend set to 'Agg'.")
except Exception as e:
    print(f"Warning: Could not set Matplotlib backend to 'Agg': {e}")
import matplotlib.pyplot as plt
from deepface import DeepFace
import pandas as pd
import base64 

RESULTS_DIR = 'test-results' 
os.makedirs(RESULTS_DIR, exist_ok=True)


SERVER_RECOGNITION_LOG_DIR = os.path.join(RESULTS_DIR, 'recognition_log_images')
os.makedirs(SERVER_RECOGNITION_LOG_DIR, exist_ok=True)


SERVER_REPORT_DIR = os.path.join(RESULTS_DIR, 'generated_reports')
os.makedirs(SERVER_REPORT_DIR, exist_ok=True)


def extract_frames_from_video(video_path, output_dir, name, max_frames=10):
    
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return [], []

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if frame_count <= 0 or fps <= 0:
        print(f"Warning: Invalid video properties for {video_path} (Frames: {frame_count}, FPS: {fps})")
        cap.release()
        return [], []

    duration = frame_count / fps

    print(f"Video: {video_path}")
    print(f"Frame Count: {frame_count}")
    print(f"FPS: {fps:.2f}")
    print(f"Duration: {duration:.2f} seconds")

    frames_to_extract = min(max_frames, frame_count)
    if frames_to_extract <= 0:
        print("No frames to extract.")
        cap.release()
        return [], []

    
    frame_indices = np.linspace(0, frame_count - 1, frames_to_extract, dtype=int)
    frame_indices = np.unique(frame_indices) 

    extracted_frame_paths = []
    face_detected_frames_data = [] 

    processed_indices = set()
    for i, frame_idx in enumerate(frame_indices):
        if frame_idx in processed_indices:
            continue
        processed_indices.add(frame_idx)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if ret:
            try:
                
                face_objs = DeepFace.extract_faces(frame,
                                                   detector_backend='opencv', 
                                                   enforce_detection=False,
                                                   align=False) 

                if face_objs and len(face_objs) > 0:
                    frame_filename = f"{name}_frame_{frame_idx}.jpg"
                    frame_path = os.path.join(output_dir, frame_filename)
                    cv2.imwrite(frame_path, frame)
                    extracted_frame_paths.append(frame_path)
                    
                    
            except Exception as e:
                
                print(f"Warning: Error processing frame {frame_idx} from {video_path}: {str(e)}")
        else:
            print(f"Warning: Could not read frame index {frame_idx} from {video_path}")


    cap.release()

    print(f"Attempted extraction for {len(frame_indices)} indices.")
    print(f"Successfully saved {len(extracted_frame_paths)} frames potentially containing faces.")
    
    return extracted_frame_paths, [] 

def measure_execution_time(func, *args, **kwargs):
    
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    execution_time = end_time - start_time
    return result, execution_time

import matplotlib.pyplot as plt
import pandas as pd 

def create_performance_visualization(metrics_df, output_path):

    
    if metrics_df.empty:
        print("Warning: No data provided for performance visualization.")
        plt.figure(figsize=(12, 8))
        plt.text(0.5, 0.5, 'Нет данных для отчета о производительности.',
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=14, color='red')
        plt.title('Отчет о производительности (Нет данных)')
        plt.axis('off')
        plt.savefig(output_path)
        plt.close()
        return output_path

    
    
    if len(metrics_df) > 1:
        plot_df = metrics_df.iloc[1:].copy() 
                                             
        print(f"Note: Excluding the first event (index {metrics_df.index[0]}) from visualization.")
    elif len(metrics_df) == 1:
        
        print("Warning: Only one data point provided. Excluding it results in no data for visualization.")
        plot_df = pd.DataFrame(columns=metrics_df.columns) 
    else:
        
        plot_df = metrics_df 

    
    if plot_df.empty:
        print("Warning: No data left for performance visualization after excluding the first event.")
        plt.figure(figsize=(12, 8))
        plt.text(0.5, 0.5, 'Нет данных для отчета о производительности\n(после исключения первого события).',
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=14, color='red')
        plt.title('Отчет о производительности (Нет данных после исключения)')
        plt.axis('off')
        plt.savefig(output_path)
        plt.close()
        return output_path

    
    plt.figure(figsize=(12, 10))

    
    plt.subplot(2, 1, 1)
    
    bars_time = plt.bar(plot_df.index, plot_df['recognition_time'], color='skyblue')
    plt.title('Время распознавания на событие (секунды) - Исключая первое событие')
    plt.ylabel('Время (с)')
    
    plt.xticks(ticks=plot_df.index, labels=plot_df['event_id'], rotation=75, ha='right', fontsize=8)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars_time:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}с', va='bottom', ha='center', fontsize=8)


    
    plt.subplot(2, 1, 2)
    
    conf_df = plot_df[plot_df['confidence'].notna()]

    if not conf_df.empty: 
        bars_conf = plt.bar(conf_df.index, conf_df['confidence'] * 100, color='lightgreen')
        for bar in bars_conf:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.1f}%', va='bottom', ha='center', fontsize=8)
    else:
         
         plt.text(0.5, 0.5, 'Нет данных о достоверности для отображаемых событий.',
                 horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes,
                 fontsize=10, color='orange')

    plt.title('Достоверность распознавания лица (%) - Исключая первое событие')
    plt.ylabel('Достоверность (%)')
    plt.ylim(0, 105)
    
    plt.xticks(ticks=plot_df.index, labels=plot_df['event_id'], rotation=75, ha='right', fontsize=8)
    plt.grid(axis='y', linestyle='--', alpha=0.7)


    plt.suptitle('Отчет о производительности распознавания лиц (Исключая первое событие)', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path)
    plt.close()

    print(f"Performance visualization (excluding first event) saved to: {output_path}")
    return output_path


def create_model_comparison_visualization(model_metrics, output_path):
    
    if not model_metrics:
         print("Warning: No data provided for model comparison.")
         
         plt.figure(figsize=(12, 8))
         plt.text(0.5, 0.5, 'Нет данных для сравнения моделей.',
                  horizontalalignment='center', verticalalignment='center',
                  fontsize=14, color='red')
         plt.title('Отчет о сравнении моделей (Нет данных)')
         plt.axis('off')
         plt.savefig(output_path)
         plt.close()
         return output_path

    models = list(model_metrics.keys())
    times = [model_metrics[model]['avg_time'] for model in models]
    
    accuracies = [model_metrics[model].get('accuracy', 0) * 100 for model in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(12, 8)) 

    
    bars1 = ax1.bar(x - width/2, times, width, label='Среднее время (с)', color='skyblue')
    ax1.set_ylabel('Среднее время распознавания (с)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.set_xlabel('Модели распознавания')
    ax1.grid(axis='y', linestyle='--', alpha=0.6, which='major', color='skyblue')

    
    for bar in bars1:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.3f}с', va='bottom', ha='center', fontsize=9, color='tab:blue')


    
    ax2 = ax1.twinx() 
    bars2 = ax2.bar(x + width/2, accuracies, width, label='Точность (%)', color='lightgreen')
    ax2.set_ylabel('Точность (%)', color='tab:green')
    ax2.tick_params(axis='y', labelcolor='tab:green')
    ax2.set_ylim(0, 105) 
    ax2.grid(axis='y', linestyle=':', alpha=0.6, which='major', color='lightgreen')

    
    for bar in bars2:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.1f}%', va='bottom', ha='center', fontsize=9, color='tab:green')


    
    fig.suptitle('Сравнение моделей распознавания лиц', fontsize=16)
    
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

    fig.tight_layout(rect=[0, 0.1, 1, 0.95]) 
    plt.savefig(output_path)
    plt.close(fig) 

    print(f"Model comparison visualization saved to: {output_path}")
    return output_path


def visualize_recognition_process(image_path, recognition_results, output_path):

    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image file: {image_path}")
            return None
        img_with_rect = img.copy()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

        if not recognition_results:
            print(f"Info: No faces provided to visualize for image {os.path.basename(image_path)}")
            
            
            
            
            pass 

        plot_title = f'Recognition Results: {os.path.basename(image_path)}'
        recognized_names = []

        for face_data in recognition_results:
            box = face_data.get('box')
            name = face_data.get('name', 'error')
            confidence = face_data.get('confidence', 0.0)
            distance = face_data.get('distance') 

            if box and len(box) == 4:
                x, y, w, h = map(int, box) 

                
                if name == "unknown":
                    color = (0, 165, 255) 
                elif name == "error":
                    color = (0, 0, 255)   
                else:
                    color = (0, 255, 0)   
                    recognized_names.append(name)

                
                cv2.rectangle(img_with_rect, (x, y), (x + w, y + h), color, 2)

                
                conf_percent = (confidence or 0.0) * 100
                
                dist_str = f", D:{distance:.2f}" if distance is not None else ""
                text = f"{name} ({conf_percent:.1f}%{dist_str})"

                
                (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                text_y = y - 10 if y - 10 > 10 else y + h + text_height + 5 

                
                cv2.rectangle(img_with_rect, (x, text_y - text_height - baseline), (x + text_width, text_y + baseline), color, cv2.FILLED)
                
                cv2.putText(img_with_rect, text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2) 

        
        if recognized_names:
            plot_title = f'Recognized: {", ".join(sorted(list(set(recognized_names))))}'
        elif recognition_results: 
             plot_title = f'Detected: {len(recognition_results)} face(s) (Unknown/Error)'


        
        plt.figure(figsize=(14, 7)) 

        
        plt.subplot(1, 2, 1)
        plt.imshow(img_rgb) 
        plt.title('Original Image')
        plt.axis('off')

        
        plt.subplot(1, 2, 2)
        img_with_rect_rgb = cv2.cvtColor(img_with_rect, cv2.COLOR_BGR2RGB)
        plt.imshow(img_with_rect_rgb)
        plt.title(plot_title)
        plt.axis('off')

        plt.suptitle(f'Recognition Detail for {os.path.basename(image_path)}', fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
        plt.savefig(output_path)
        plt.close()

        
        return output_path

    except Exception as e:
        print(f"Error during visualization for {image_path}: {e}", exc_info=True)
        return None


def image_to_base64(image_path):
    
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image {image_path} to Base64: {e}")
        return None


if __name__ == "__main__":
    print("Test Utilities Module (test_utils.py) Loaded.")
    print(f"Default results directory: {os.path.abspath(RESULTS_DIR)}")
    print(f"Default server log image directory: {os.path.abspath(SERVER_RECOGNITION_LOG_DIR)}")
    print(f"Default server report directory: {os.path.abspath(SERVER_REPORT_DIR)}")
    
    
    
    
    
    
    
    
    
    