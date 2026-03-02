import json
import os
from collections import defaultdict
from pathlib import Path

CONFIDENCE_THRESHOLDS = [0.2, 0.5, 0.8]

def print_statistics(threshold):
    """Выводит статистику детекций для заданного порога"""
    json_file = f"analysis/detections_threshold_{threshold}.json"
    
    if not os.path.exists(json_file):
        print(f"Файл {json_file} не найден!")
        return
    
    with open(json_file, 'r') as f:
        detections = json.load(f)
    
    total_detections = sum(len(det) for det in detections.values())
    
    # Подсчет объектов по классам
    class_counts = defaultdict(int)
    for image_detections in detections.values():
        for detection in image_detections:
            class_counts[detection['class']] += 1
    
    # Расчет средней уверенности
    all_confidences = []
    for image_detections in detections.values():
        for detection in image_detections:
            all_confidences.append(detection['confidence'])
    
    avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
    
    print(f"\n{'='*80}")
    print(f"СТАТИСТИКА ДЛЯ ПОРОГА УВЕРЕННОСТИ: {threshold}")
    print(f"{'='*80}")
    print(f"Всего обнаружено объектов: {total_detections}")
    print(f"Средняя уверенность: {avg_confidence:.4f}")
    print(f"\nОбваружено по классам:")
    for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {class_name:20s}: {count:5d}")
    
    # Выводим статистику по изображениям
    detections_per_image = [len(det) for det in detections.values()]
    print(f"\nСтатистика по изображениям:")
    print(f"  Минимум объектов на изображение: {min(detections_per_image) if detections_per_image else 0}")
    print(f"  Максимум объектов на изображение: {max(detections_per_image) if detections_per_image else 0}")
    print(f"  Средное количество объектов:     {sum(detections_per_image)/len(detections_per_image) if detections_per_image else 0:.2f}")
    print(f"  Всего проанализированных изображений: {len(detections)}")

def compare_thresholds():
    """Сравнивает результаты для разных порогов"""
    print(f"\n{'='*80}")
    print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ ПО РАЗНЫМ ПОРОГАМ")
    print(f"{'='*80}")
    print(f"{'Порог':<10} {'Всего объектов':<20} {'Средняя уверенность':<20}")
    print("-" * 50)
    
    for threshold in CONFIDENCE_THRESHOLDS:
        json_file = f"analysis/detections_threshold_{threshold}.json"
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                detections = json.load(f)
            
            total = sum(len(det) for det in detections.values())
            
            all_confidences = []
            for image_detections in detections.values():
                for detection in image_detections:
                    all_confidences.append(detection['confidence'])
            avg_conf = sum(all_confidences) / len(all_confidences) if all_confidences else 0
            
            print(f"{threshold:<10.1f} {total:<20d} {avg_conf:<20.4f}")
    

def load_annotations(path="annotations.json"):
    """Загружает разметку из JSON файла со структурой {image: [bboxes...]}.
    Пустые списки (никаких объектов) отбрасываются, чтобы не дискредитировать
    precision/recall, и выводится сообщение.
    """
    if not os.path.exists(path):
        print("Файл разметки не найден, расчёт метрик пропущен.")
        return None
    with open(path, 'r') as f:
        ann = json.load(f)
    orig = len(ann)
    ann = {k: v for k, v in ann.items() if v}
    if len(ann) != orig:
        print(f"Отфильтровано {orig-len(ann)} изображений без объектов в разметке")
    return ann


def compute_iou(box1, box2):
    # boxes dict с ключами x1,y1,x2,y2
    xa = max(box1['x1'], box2['x1'])
    ya = max(box1['y1'], box2['y1'])
    xb = min(box1['x2'], box2['x2'])
    yb = min(box1['y2'], box2['y2'])
    inter_w = max(0, xb - xa)
    inter_h = max(0, yb - ya)
    inter_area = inter_w * inter_h
    area1 = (box1['x2'] - box1['x1']) * (box1['y2'] - box1['y1'])
    area2 = (box2['x2'] - box2['x1']) * (box2['y2'] - box2['y1'])
    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0


def evaluate_precision_recall(annotations, detections, iou_threshold=0.5):
    """Оценка precision/recall для одного порога.
    Возвращает tuple (precision, recall, TP, FP, FN).
    annotations: dict image->list of gt boxes (dict)
    detections: dict image->list of det boxes (dict) как из анализатора
    """
    TP = 0
    FP = 0
    FN = 0
    for img, gt_list in annotations.items():
        det_list = detections.get(img, [])
        matched = [False] * len(gt_list)
        # для каждого детекта найдем лучший gt
        for det in det_list:
            # match the detection to any ground truth by IoU (ignore class mismatch)
            best_iou = 0
            best_idx = -1
            for idx, gt in enumerate(gt_list):
                iou = compute_iou(gt['bbox'], det['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_iou >= iou_threshold and not (best_idx == -1 or matched[best_idx]):
                TP += 1
                matched[best_idx] = True
            else:
                FP += 1
        # не совпавшие gt считаются пропусками
        FN += sum(1 for m in matched if not m)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    return precision, recall, TP, FP, FN


def compute_metrics_for_thresholds(annotation_file="annotations.json"):
    ann = load_annotations(annotation_file)
    if ann is None:
        return

    # basic summary of annotations
    total_gt = sum(len(v) for v in ann.values())
    print(f"\nЗагружено {len(ann)} изображений разметки, всего {total_gt} объектов")
    if total_gt < 10:
        print("[!]: Разметка слишком мала для осмысленных метрик. Добавьте больше данных.")

    print(f"\n{'='*80}")
    print("ПРИЦИОН И РЕКОЛЛ ПО ПОРОГАМ")
    print(f"{'='*80}")
    for threshold in CONFIDENCE_THRESHOLDS:
        json_file = f"analysis/detections_threshold_{threshold}.json"
        if not os.path.exists(json_file):
            continue
        with open(json_file, 'r') as f:
            dets = json.load(f)
        prec, rec, tp, fp, fn = evaluate_precision_recall(ann, dets)
        print(f"Порог {threshold:<4.1f}: precision={prec:.3f}, recall={rec:.3f}")
        print(f"  TP={tp}, FP={fp}, FN={fn}  (ложных срабатываний={fp})")


def list_sample_detections(threshold, num_samples=3):
    """Выводит примеры детекций"""
    json_file = f"analysis/detections_threshold_{threshold}.json"
    if not os.path.exists(json_file):
        return
    
    with open(json_file, 'r') as f:
        detections = json.load(f)
    
    print(f"\nПримеры обнаруженных объектов (порог {threshold}):")
    count = 0
    for image_name, image_detections in list(detections.items())[:num_samples]:
        if image_detections:
            print(f"\n  {image_name}:")
            for detection in image_detections[:3]:
                print(f"    - {detection['class']:15s} | confidence: {detection['confidence']:.3f}")
            count += 1
            if count >= num_samples:
                break

if __name__ == "__main__":
    # Выводим статистику для каждого порога
    for threshold in CONFIDENCE_THRESHOLDS:
        print_statistics(threshold)
    
    # Сравниваем результаты
    compare_thresholds()
    
    # Выводим примеры
    for threshold in CONFIDENCE_THRESHOLDS:
        list_sample_detections(threshold, num_samples=2)
    
    # если есть файл разметки, рассчитаем precision/recall
    compute_metrics_for_thresholds()
    
    print(f"\n{'='*80}")
    print("АНАЛИЗ ЗАВЕРШЕН")
    print(f"{'='*80}")
