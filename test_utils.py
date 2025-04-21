import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from deepface import DeepFace
import pandas as pd

RESULTS_DIR = '/test-results' 
os.makedirs(RESULTS_DIR, exist_ok=True)

def extract_frames_from_video(video_path, output_dir, name, max_frames=10):
    
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps
    
    print(f"Видео: {video_path}")
    print(f"Количество кадров: {frame_count}")
    print(f"FPS: {fps}")
    print(f"Длительность: {duration:.2f} секунд")
    
    frames_to_extract = min(max_frames, frame_count)
    frame_indices = np.linspace(0, frame_count-1, frames_to_extract, dtype=int)
    
    extracted_frames = []
    face_detected_frames = []
    
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            try:
                face_objs = DeepFace.extract_faces(frame, detector_backend='opencv')
                if len(face_objs) > 0:
                    frame_path = os.path.join(output_dir, f"{name}_frame_{i}.jpg")
                    cv2.imwrite(frame_path, frame)
                    extracted_frames.append(frame_path)
                    face_detected_frames.append(frame)
            except Exception as e:
                print(f"Ошибка при обработке кадра {i}: {str(e)}")
    
    cap.release()
    
    print(f"Извлечено {len(extracted_frames)} кадров с лицами")
    return extracted_frames, face_detected_frames

def measure_execution_time(func, *args, **kwargs):
    
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    execution_time = end_time - start_time
    return result, execution_time

def create_performance_visualization(metrics_df, output_path):
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.bar(metrics_df['name'], metrics_df['recognition_time'], color='skyblue')
    plt.title('Время распознавания лиц (секунды)')
    plt.ylabel('Время (с)')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    
    plt.subplot(2, 1, 2)
    plt.bar(metrics_df['name'], metrics_df['confidence'], color='lightgreen')
    plt.title('Уверенность распознавания (%)')
    plt.ylabel('Уверенность (%)')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    return output_path


def create_model_comparison_visualization(model_metrics, output_path):
    
    models = list(model_metrics.keys())
    times = [model_metrics[model]['avg_time'] for model in models]
    accuracies = [model_metrics[model]['accuracy'] * 100 for model in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax1 = plt.figure(figsize=(12, 8)), plt.subplot(111)
    
    
    bars1 = ax1.bar(x - width/2, times, width, label='Время (с)', color='skyblue')
    ax1.set_ylabel('Время (с)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, accuracies, width, label='Точность (%)', color='lightgreen')
    ax2.set_ylabel('Точность (%)', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45)
    ax1.set_title('Сравнение моделей распознавания лиц')
    ax1.legend(handles=[bars1, bars2], loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    return output_path


def visualize_recognition_process(image_path, detected_face, name, confidence, output_path):
    
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    
    face_objs = DeepFace.extract_faces(img, detector_backend='opencv', enforce_detection=False)
    if len(face_objs) > 0:
        face_obj = face_objs[0]
        facial_area = face_obj['facial_area']
        
        x = facial_area['x']
        y = facial_area['y']
        w = facial_area['w']
        h = facial_area['h']
        
        
        img_with_rect = img.copy()
        cv2.rectangle(img_with_rect, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        
        confidence_text = f"{name}: {confidence:.2f}%"
        cv2.putText(img_with_rect, confidence_text, (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        
        plt.figure(figsize=(12, 6))
        
        
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title('Исходное изображение')
        plt.axis('off')
        
        
        plt.subplot(1, 2, 2)
        plt.imshow(img_with_rect)
        plt.title(f'Распознано: {confidence_text}')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        return output_path
    
    return None

if __name__ == "__main__":
    print("Модуль для тестирования программы распознавания лиц загружен успешно.")
