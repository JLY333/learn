import torch
import cv2
import os
from ultralytics import YOLO
from torchvision import models, transforms
from PIL import Image
import numpy as np

# -----------------------------
# 配置
# -----------------------------
IMAGE_PATH = r"dataset\images\test\inclusion_241_jpg.rf.b2d770d3314649cc095e60cea8d53b76.jpg"  # 原图
YOLO_WEIGHTS = r"runs\detect\train\weights\best.pt"
RESNET_WEIGHTS = "resnet18_defect_cls.pth"
CLS_TRAIN_DIR = "cls_dataset/train"  # 用于读取类别名
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# 读取分类类别顺序（必须和训练时一致）
# -----------------------------
class_names = sorted(os.listdir(CLS_TRAIN_DIR))
print("ResNet Classes:", class_names)

# -----------------------------
# 加载 YOLO
# -----------------------------
yolo = YOLO(YOLO_WEIGHTS)

# -----------------------------
# 加载 ResNet18
# -----------------------------
resnet = models.resnet18(pretrained=False)
resnet.fc = torch.nn.Linear(resnet.fc.in_features, len(class_names))
resnet.load_state_dict(torch.load(RESNET_WEIGHTS, map_location=DEVICE))
resnet.to(DEVICE)
resnet.eval()

# -----------------------------
# ResNet 预处理（和训练保持一致）
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# 推理
# -----------------------------
img_bgr = cv2.imread(IMAGE_PATH)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

results = yolo(IMAGE_PATH, conf=0.25)
result = results[0]

if result.boxes is None or len(result.boxes) == 0:
    print("❌ YOLO 未检测到缺陷")
    exit()

for box in result.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    roi = img_rgb[y1:y2, x1:x2]
    if roi.size == 0:
        continue

    roi_pil = Image.fromarray(roi)
    roi_tensor = transform(roi_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = resnet(roi_tensor)
        prob = torch.softmax(logits, dim=1)
        cls_id = prob.argmax(dim=1).item()
        cls_conf = prob[0, cls_id].item()

    cls_name = class_names[cls_id]

    label = f"{cls_name} ({cls_conf:.2f})"
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img_bgr, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

cv2.imshow("YOLO + ResNet18", img_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
