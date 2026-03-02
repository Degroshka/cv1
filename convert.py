# конвертер YOLO‑txt → annotations.json
import json, glob, os

out = {}
# учитываем метки во всех подпапках (train/valid/test и т.д.)
for txt in glob.glob("**/labels/**/*.txt", recursive=True):
    # сопоставляем путь метки пути к изображению: заменяем сегмент "labels" на "images"
    imgfile = txt.replace(os.sep + "labels" + os.sep, os.sep + "images" + os.sep)
    imgfile = os.path.splitext(imgfile)[0] + ".jpg"  # поменяйте на .png, если у вас другой формат

    # если файл не существует, попробуем поиск по basename в любом месте
    if not os.path.exists(imgfile):
        basename = os.path.basename(imgfile)
        # искать в одной из директорий images, train/images, valid/images, test/images
        for base in ["images", "train/images", "valid/images", "test/images"]:
            candidate = os.path.join(base, basename)
            if os.path.exists(candidate):
                imgfile = candidate
                break
    # запомним ключ как относительный путь от корня рабочей папки
    relimg = os.path.relpath(imgfile)
    # use forward slashes in JSON keys for consistency
    relimg = relimg.replace(os.sep, "/")
    # определим размеры изображения для перевода нормированных координат
    try:
        import cv2
        img = cv2.imread(imgfile)
        if img is None:
            raise ValueError("imread вернул None")
        h, w = img.shape[:2]
    except Exception as e:
        print(f"ошибка при открытии {imgfile}: {e}")
        h, w = 1, 1

    boxes = []
    with open(txt) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue  # пустая строка
            if len(parts) < 5:
                print(f"пропускаю некорректную строку в {txt}: {parts}")
                continue
            cls, x, y, nw, nh = map(float, parts[:5])
            # нормированные XYWH → абсолютные x1,y1,x2,y2
            abs_w = nw * w
            abs_h = nh * h
            cx = x * w
            cy = y * h
            x1 = cx - abs_w / 2
            y1 = cy - abs_h / 2
            x2 = cx + abs_w / 2
            y2 = cy + abs_h / 2
            bbox = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
            boxes.append({"class": int(cls), "bbox": bbox})
    # добавляем запись только если есть хоть один бокс
    if boxes:
        out[relimg] = boxes
with open("annotations.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)