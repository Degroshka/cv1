from ultralytics import YOLO
import glob
import os
import cv2
import json
import numpy as np
from pathlib import Path

model = YOLO("yolov8n.pt")

# Создаем папки для результатов
os.makedirs("results", exist_ok=True)
# создадим папки для каждой границы уверенности
CONFIDENCE_THRESHOLDS = [0.2, 0.5, 0.8]
for thr in CONFIDENCE_THRESHOLDS:
    os.makedirs(f"results/threshold_{thr}", exist_ok=True)

os.makedirs("analysis", exist_ok=True)
os.makedirs("detections_json", exist_ok=True)


def extract_detections_by_threshold(result, threshold):
    """Извлекает детекции для конкретного порога"""
    detections = []
    for box in result.boxes:
        conf = float(box.conf[0].item())
        if conf >= threshold:  # Фильтруем по порогу
            x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
            class_id = int(box.cls[0].item())
            class_name = result.names[class_id]
            
            detections.append({
                "class": class_name,
                "class_id": class_id,
                "confidence": conf,
                "bbox": {
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "width": x2 - x1,
                    "height": y2 - y1
                }
            })
    return detections

def save_visualization_with_threshold(image_path, result, output_path, threshold):
    """Сохраняет изображение с bounding boxes и confidence scores для конкретного порога"""
    img = cv2.imread(image_path)
    
    for box in result.boxes:
        conf = float(box.conf[0].item())
        if conf >= threshold:  # Фильтруем визуализацию по порогу
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0].item())
            class_name = result.names[class_id]
            
            # Рисуем bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Рисуем текст с классом и confidence
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 255, 0), 2)
    
    cv2.imwrite(output_path, img)

# Словарь для хранения результатов по порогам
results_by_threshold = {threshold: {} for threshold in CONFIDENCE_THRESHOLDS}

print("=" * 80)
print("ЗАПУСК ДЕТЕКЦИИ ОБЪЕКТОВ")
print("=" * 80)

# рекурсивно собираем пути ко всем изображениям в любом каталоге images
image_paths = glob.glob("**/images/*.*", recursive=True)
print(f"\nНайдено изображений: {len(image_paths)}")

for idx, path in enumerate(image_paths, 1):
    if idx % 100 == 0:
        print(f"Обработано: {idx}/{len(image_paths)}")
    # относительный путь от корня рабочего каталога используется как ключ
    relname = os.path.relpath(path)
    # normalize to forward slashes so annotations keys match
    relname = relname.replace(os.sep, "/")
    
    # Запускаем детекцию с низким порогом (потом будем фильтровать)
    result = model(path, conf=0.1)[0]
    
    thresholds_with_dets = []
    for threshold in CONFIDENCE_THRESHOLDS:
        det_list = extract_detections_by_threshold(result, threshold)
        results_by_threshold[threshold][relname] = det_list
        if det_list:
            thresholds_with_dets.append(threshold)
    
    # выберем наибольший порог, который удовлетворяется хотя бы одной детекцией
    if thresholds_with_dets:
        selected_thr = max(thresholds_with_dets)
        # reconstruct directory in results using forward‑slash path components
        subdir = os.path.dirname(relname).replace("/", os.sep)
        outdir = os.path.join("results", f"threshold_{selected_thr}", subdir)
        os.makedirs(outdir, exist_ok=True)
        output_path = os.path.join(outdir, os.path.basename(relname))
        save_visualization_with_threshold(path, result, output_path, selected_thr)
    else:
        os.makedirs("results/threshold_none", exist_ok=True)
        output_path = os.path.join("results/threshold_none", os.path.basename(relname))
        from shutil import copyfile
        copyfile(path, output_path)

# Сохраняем JSON результаты
for threshold, detections_dict in results_by_threshold.items():
    output_file = f"analysis/detections_threshold_{threshold}.json"
    with open(output_file, 'w') as f:
        json.dump(detections_dict, f, indent=2)

print(f"\nОбработано всех изображений: {len(image_paths)}")
# сохранить json результаты
for threshold, detections_dict in results_by_threshold.items():
    output_file = f"analysis/detections_threshold_{threshold}.json"
    with open(output_file, 'w') as f:
        json.dump(detections_dict, f, indent=2)

print("\nРезультаты сохранены в папках:")
print("  - results/ (визуализация разбита по threshold_X и threshold_none)")
print("  - analysis/ (JSON с детекциями)")
print("\n" + "=" * 80)