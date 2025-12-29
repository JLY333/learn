
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2

# 加载训练好的 YOLOv8 模型
model = YOLO(r"runs\detect\train\weights\best.pt")  # 使用训练好的模型权重（确保路径正确）

# 图像路径
img_path = "dataset/images/val/crazing_241_jpg.rf.17c031adb49a4138cd3ce5e1aa903726.jpg"  # 替换为你的图像路径

# 调整置信度阈值，确保低置信度的框也会被保留
results = model(img_path, conf=0.1)  # 设置较低的置信度阈值（如0.3）

# 获取推理结果的第一个元素（如果有多个图像）
result = results[0]

# 打印检测框信息
print("Detected Boxes:", result.boxes)  # 输出检测框的详细信息（包括坐标、类别、置信度）

# 如果检测到目标，使用 plot() 方法绘制带有检测框的图像
if len(result.boxes) > 0:
    result.plot()  # 绘制检测框和标签

    # 显示带有检测框的图像
    plt.imshow(cv2.cvtColor(result.orig_img, cv2.COLOR_BGR2RGB))  # 将 BGR 转换为 RGB 显示
    plt.axis('off')  # 关闭坐标轴
    plt.show()
else:
    print("No detections found!")

# 如果你想保存图像
result.save()  # 保存图像到 runs/detect/ 文件夹（包含检测框）









