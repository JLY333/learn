import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # 数据路径
    # -------------------------
    data_root = "cls_dataset"   # 里面有 train / val

    # -------------------------
    # 数据增强
    # -------------------------
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_set = datasets.ImageFolder(os.path.join(data_root, "train"), transform=train_tf)
    val_set   = datasets.ImageFolder(os.path.join(data_root, "val"), transform=val_tf)

    train_loader = DataLoader(
        train_set,
        batch_size=32,
        shuffle=True,
        num_workers=0   # 🔴 Windows 必须这样
    )

    val_loader = DataLoader(
        val_set,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    num_classes = len(train_set.classes)
    print("Classes:", train_set.classes)

    # -------------------------
    # 模型
    # -------------------------
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # -------------------------
    # 训练
    # -------------------------
    epochs = 25
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {total_loss/len(train_loader):.4f}")

        # -------- 验证 --------
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = correct / total * 100
        print(f"Validation Accuracy: {acc:.2f}%")

    torch.save(model.state_dict(), "resnet18_defect_cls.pth")
    print("Training finished.")

# 🔴 Windows 必须要这个
if __name__ == "__main__":
    main()


