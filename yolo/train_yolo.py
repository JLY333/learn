from ultralytics import YOLO
import torch

# --- 可选：设置多进程启动方式（针对Windows推荐）---
# 如果你不加这句有时也会报错，显式设置为 'spawn' 更稳妥
if __name__ == '__main__':
    # 1. 防止打包成exe出错（非必须，但无害）
    # from multiprocessing import freeze_support
    # freeze_support()
    
    # 2. 显式设置多进程启动方式（针对Windows的关键修复）
    torch.multiprocessing.set_start_method('spawn', force=True) 

    # 3. 加载模型
    model = YOLO("yolov8n.pt")  # 加载预训练模型
    
    # 4. 开始训练
    # 关键参数调整: workers=0 或 1, persistent_workers=False
    results = model.train(
        data="dataset/data.yaml",    # 数据集配置文件路径
        epochs=20,                   # 训练轮数
        imgsz=640,                   # 图像尺寸
        
        # --- 针对 Windows 报错的修复参数 ---
        workers=1,                   # 数据加载线程数。设为 0 也可以（最慢但最稳），1-2 通常平衡较好
        batch=16,                    # 根据你的显存调整，如果显存不够请调小（如 8, 4）
        
        # --- 可选优化 ---
        # project='runs/train',      # 可以指定项目保存路径
        # name='exp'                 # 实验名称
    )
    
    # 5. 训练完成后可以进行验证（自动验证，无需手动调用 model.val()）
    # YOLOv8 已经集成了训练后自动进行验证
    print("Training completed!")
