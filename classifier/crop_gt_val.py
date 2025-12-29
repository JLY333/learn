import os
import cv2

IMG_DIR = "dataset/images/val"
LABEL_DIR = "dataset/labels/val"
OUT_DIR = "cls_dataset/val"

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]

os.makedirs(OUT_DIR, exist_ok=True)

for cls in CLASS_NAMES:
    os.makedirs(os.path.join(OUT_DIR, cls), exist_ok=True)

for img_name in os.listdir(IMG_DIR):
    if not img_name.endswith(".jpg"):
        continue

    img_path = os.path.join(IMG_DIR, img_name)
    label_path = os.path.join(
        LABEL_DIR, img_name.replace(".jpg", ".txt")
    )

    if not os.path.exists(label_path):
        continue

    img = cv2.imread(img_path)
    h, w, _ = img.shape

    with open(label_path) as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        cls_id, xc, yc, bw, bh = map(float, line.split())
        cls_id = int(cls_id)

        # YOLO → 像素坐标
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        cls_name = CLASS_NAMES[cls_id]
        out_name = f"{img_name[:-4]}_{i}.jpg"
        cv2.imwrite(
            os.path.join(OUT_DIR, cls_name, out_name),
            crop
        )