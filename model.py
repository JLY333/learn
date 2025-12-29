import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchviz import make_dot
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from scipy.stats import kurtosis, skew
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pyvis.network import Network
import warnings
warnings.filterwarnings("ignore")

# ===================== 1. 核心参数配置 =====================
DATA_PATH = "./JNU_轴承故障_600rpm.csv"
MODEL_SAVE_PATH = "./best_bearing_bp_model.pth"
LOG_SAVE_PATH = "./bearing_bp_logs"
CM_SAVE_PATH = "./confusion_matrix.png"
NETWORK_VIS_PATH = "./bp_network_visual.html"
WEIGHT_VIS_PATH = "./weight_distribution.png"

WINDOW_SIZE = 1024
STEP = 512
INPUT_DIM = 8       
HIDDEN_DIM1 = 128   
HIDDEN_DIM2 = 64    
OUTPUT_DIM = 4      
BATCH_SIZE = 32     
LEARNING_RATE = 1e-3
EPOCHS = 50
PATIENCE = 5        
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 训练设备：{DEVICE}")
print(f"🔧 PyTorch版本：{torch.__version__}")

# ===================== 2. 自定义数据集类 =====================
class BearingDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.features):
            raise IndexError(f"❌ 索引{idx}超出范围（总样本数：{len(self.features)}）")
        return self.features[idx], self.labels[idx]

# ===================== 3. 数据预处理函数 =====================
def preprocess_bearing_data(data_path, window_size, step):
    try:
        df = pd.read_csv(data_path, encoding="utf-8")
        print("✅ 数据集编码：utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(data_path, encoding="gbk")
        print("✅ 数据集编码：gbk（Windows默认）")
    except Exception as e:
        raise FileNotFoundError(f"❌ 加载数据集失败：{e}")
    
    drop_cols = ["转速(rpm)", "源文件"]
    dropped_cols = [col for col in drop_cols if col in df.columns]
    df = df.drop(columns=dropped_cols)
    print(f"✅ 删除冗余列：{dropped_cols}")
    
    required_cols = ["振动信号", "故障标签"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"❌ 数据集缺少核心列：{missing_cols}")
    
    print(f"✅ 原始数据集总行数：{len(df)}")
    print(f"✅ 故障标签分布：{df['故障标签'].value_counts().sort_index().to_dict()}")
    
    all_features = []
    all_labels = []
    for label in [0, 1, 2, 3]:
        label_data = df[df["故障标签"] == label]["振动信号"].values
        if len(label_data) < window_size:
            print(f"⚠️ 标签{label}数据长度不足，跳过")
            continue
        
        for i in range(0, len(label_data) - window_size + 1, step):
            window_signal = label_data[i:i+window_size]
            if isinstance(window_signal[0], str):
                window_signal = np.array([eval(signal) for signal in window_signal])
            else:
                window_signal = np.array(window_signal, dtype=np.float32)
            
            feature = [
                np.mean(window_signal), np.std(window_signal), kurtosis(window_signal), skew(window_signal),
                np.max(np.abs(window_signal)), np.sqrt(np.mean(window_signal**2)),
                np.max(np.abs(window_signal)) / np.sqrt(np.mean(window_signal**2)),
                np.max(np.abs(window_signal)) / np.mean(np.abs(window_signal))
            ]
            all_features.append(feature)
            all_labels.append(label)
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(all_features)
    print(f"\n📊 数据预处理完成：")
    print(f"   分窗后总样本数：{len(features_scaled)}")
    print(f"   各标签样本数：{np.bincount(all_labels)}")
    return features_scaled, all_labels

# ===================== 4. BP神经网络模型（记录激活值） =====================
class BearingBPNet(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(BearingBPNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.activations = {}
    
    def forward(self, x):
        x1 = self.relu(self.fc1(x))
        self.activations['hidden1'] = x1.detach().cpu().numpy()
        x1 = self.dropout(x1)
        
        x2 = self.relu(self.fc2(x1))
        self.activations['hidden2'] = x2.detach().cpu().numpy()
        x2 = self.dropout(x2)
        
        x3 = self.fc3(x2)
        self.activations['output'] = x3.detach().cpu().numpy()
        return x3
    
    def reset_activations(self):
        self.activations = {}

# ===================== 5. 可视化函数合集（核心修复序列化问题） =====================
def visualize_network_structure(model, dummy_input, save_path):
    """
    修复点：
    1. 强制将numpy类型转为Python原生int，解决JSON序列化问题
    2. 简化节点采样逻辑，避免numpy类型混入
    3. 计算图提示优化
    """
    # -------- 1. 计算图可视化（非必需，失败仅提示） --------
    try:
        model_cpu = model.to("cpu")
        dummy_input_cpu = dummy_input.to("cpu")
        y = model_cpu(dummy_input_cpu)
        dot = make_dot(y, params=dict(model_cpu.named_parameters()))
        dot.render("bp_computation_graph", format="pdf")
        print(f"✅ 计算图已保存为：bp_computation_graph.pdf")
        model.to(DEVICE)
    except Exception as e:
        print(f"⚠️ 计算图绘制失败（非必需，可忽略）：{e}")
        print(f"   💡 解决方法：安装Graphviz系统库并配置PATH，或忽略该提示")
    
    # -------- 2. 交互式网络拓扑可视化（修复JSON序列化） --------
    net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white", directed=True)
    net.barnes_hut()
    
    layers = {
        "Input (输入层)": INPUT_DIM,
        "Hidden 1 (隐藏层1)": HIDDEN_DIM1,
        "Hidden 2 (隐藏层2)": HIDDEN_DIM2,
        "Output (输出层)": OUTPUT_DIM
    }
    layer_colors = ["#00ff00", "#0080ff", "#ff8000", "#ff0000"]
    layer_ids = []
    node_id = 0  # 用Python原生int，避免numpy类型
    
    # 添加神经元节点（纯Python int，无numpy）
    for i, (layer_name, num_neurons) in enumerate(layers.items()):
        layer_nodes = []
        # 采样减少节点数（用Python random，避免numpy）
        sample_num = min(10, num_neurons)
        # 生成采样索引（Python原生list）
        sample_indices = list(range(sample_num))
        
        for n in sample_indices:
            neuron_name = f"{layer_name}-{n+1}"
            # 节点ID用原生int
            net.add_node(node_id, label=neuron_name, color=layer_colors[i], size=15)
            layer_nodes.append(node_id)
            node_id += 1
        layer_ids.append(layer_nodes)
        
        # 添加层标签（直接指定位置，原生int）
        net.add_node(
            node_id, 
            label=layer_name, 
            color="#ffffff", 
            size=30, 
            shape="box",
            x=int(i*200),  # 强制转原生int
            y=int(-100)    # 强制转原生int
        )
        node_id += 1
    
    # 添加层间连接（纯Python类型，避免numpy）
    for i in range(len(layer_ids)-1):
        current_layer = layer_ids[i]
        next_layer = layer_ids[i+1]
        
        # 手动采样（避免numpy.random生成int32）
        sample_size_current = min(5, len(current_layer))
        sample_size_next = min(5, len(next_layer))
        # 用Python random.sample，返回原生int列表
        import random
        random.seed(42)  # 固定种子，保证结果一致
        sample_current = random.sample(current_layer, sample_size_current)
        sample_next = random.sample(next_layer, sample_size_next)
        
        for c_node in sample_current:
            for n_node in sample_next:
                # 边的属性全部用原生类型
                net.add_edge(
                    int(c_node), int(n_node),  # 强制转原生int
                    color="#888888", 
                    width=float(0.5)  # 强制转原生float
                )
    
    # 保存HTML（修复序列化）
    try:
        net.save_graph(save_path)
        print(f"✅ 交互式网络拓扑已保存为：{save_path}（用浏览器打开查看）")
    except Exception as e:
        print(f"⚠️ 交互式拓扑生成失败：{e}")
        print(f"   💡 已跳过该可视化，不影响模型训练和其他可视化功能")

def visualize_weight_distribution(model, save_path):
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    params = {
        "fc1_weight": model.fc1.weight.data.cpu().numpy().flatten(),
        "fc1_bias": model.fc1.bias.data.cpu().numpy().flatten(),
        "fc2_weight": model.fc2.weight.data.cpu().numpy().flatten(),
        "fc2_bias": model.fc2.bias.data.cpu().numpy().flatten(),
        "fc3_weight": model.fc3.weight.data.cpu().numpy().flatten(),
        "fc3_bias": model.fc3.bias.data.cpu().numpy().flatten()
    }
    
    for i, (name, data) in enumerate(params.items()):
        axes[i].hist(data, bins=50, alpha=0.7, color="#1f77b4")
        axes[i].set_title(f"{name} 分布", fontsize=12)
        axes[i].set_xlabel("参数值", fontsize=10)
        axes[i].set_ylabel("频次", fontsize=10)
        axes[i].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"✅ 权重/偏置分布已保存为：{save_path}")

def visualize_activations(model, sample_data, save_prefix="./activation_"):
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    model.eval()
    with torch.no_grad():
        _ = model(sample_data.to(DEVICE))
    
    for layer_name, activations in model.activations.items():
        sample_act = activations[0]
        plt.figure(figsize=(8, 4))
        plt.hist(sample_act, bins=30, alpha=0.7, color="#ff7f0e")
        plt.title(f"{layer_name} 层神经元激活值分布", fontsize=12)
        plt.xlabel("激活值", fontsize=10)
        plt.ylabel("神经元数量", fontsize=10)
        plt.grid(alpha=0.3)
        save_path = f"{save_prefix}{layer_name}.png"
        plt.savefig(save_path, dpi=300)
        plt.show()
        print(f"✅ {layer_name}层激活值分布已保存为：{save_path}")
    model.reset_activations()

def plot_confusion_matrix(true_labels, pred_labels, save_path):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["正常", "内圈故障", "滚动体故障", "外圈故障"],
                yticklabels=["正常", "内圈故障", "滚动体故障", "外圈故障"])
    plt.xlabel("预测标签", fontsize=12)
    plt.ylabel("真实标签", fontsize=12)
    plt.title("轴承故障诊断混淆矩阵", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"📊 混淆矩阵已保存至：{save_path}")

# ===================== 6. 训练函数 =====================
def train_bp_model(model, train_loader, test_loader, criterion, optimizer, epochs, patience, device):
    model.to(device)
    best_test_acc = 0.0
    patience_counter = 0
    writer = SummaryWriter(LOG_SAVE_PATH)
    
    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * features.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        model.eval()
        test_loss, test_correct, test_total = 0.0, 0, 0
        with torch.no_grad():
            for features, labels in test_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item() * features.size(0)
                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        
        train_acc = train_correct / train_total
        test_acc = test_correct / test_total
        avg_train_loss = train_loss / train_total
        avg_test_loss = test_loss / test_total
        
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Loss/Test", avg_test_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("Accuracy/Test", test_acc, epoch)
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            patience_counter = 0
            print(f"📌 Epoch {epoch+1} | 测试准确率提升至 {best_test_acc:.4f} | 保存最优模型")
        else:
            patience_counter += 1
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Test Loss: {avg_test_loss:.4f} | Test Acc: {test_acc:.4f} | "
                  f"早停计数器：{patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"\n⚠️ 早停触发：连续{patience}轮测试准确率无提升，终止训练")
                break
    
    writer.close()
    print(f"\n🎯 训练结束 | 最佳测试准确率：{best_test_acc:.4f}")
    return model

# ===================== 7. 主流程 =====================
if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        print(f"\n❌ 找不到数据集文件！当前路径：{DATA_PATH}")
        exit()
    
    print("\n===== 步骤1：数据预处理 =====")
    try:
        features, labels = preprocess_bearing_data(DATA_PATH, WINDOW_SIZE, STEP)
    except Exception as e:
        print(f"❌ 数据预处理失败：{e}")
        exit()
    
    print("\n===== 步骤2：划分训练/测试集 =====")
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.3, random_state=42, stratify=labels
    )
    print(f"✅ 训练集样本数：{len(X_train)} | 测试集样本数：{len(X_test)}")
    
    print("\n===== 步骤3：构建数据加载器 =====")
    train_dataset = BearingDataset(X_train, y_train)
    test_dataset = BearingDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print("\n===== 步骤4：初始化模型并可视化结构 =====")
    model = BearingBPNet(INPUT_DIM, HIDDEN_DIM1, HIDDEN_DIM2, OUTPUT_DIM)
    dummy_input = torch.randn(1, INPUT_DIM)  # 纯CPU张量，避免设备冲突
    visualize_network_structure(model, dummy_input, NETWORK_VIS_PATH)
    print(f"✅ 模型结构：\n{model}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    print("\n===== 步骤5：开始训练BP神经网络 =====")
    trained_model = train_bp_model(
        model, train_loader, test_loader, criterion, optimizer, EPOCHS, PATIENCE, DEVICE
    )
    
    print("\n===== 步骤6：可视化权重/偏置分布 =====")
    best_model = BearingBPNet(INPUT_DIM, HIDDEN_DIM1, HIDDEN_DIM2, OUTPUT_DIM)
    best_model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    best_model.to(DEVICE)
    visualize_weight_distribution(best_model, WEIGHT_VIS_PATH)
    
    print("\n===== 步骤7：可视化神经元激活值 =====")
    sample_features, _ = next(iter(test_loader))
    visualize_activations(best_model, sample_features[:1], save_prefix="./activation_")
    
    print("\n===== 步骤8：验证最优模型 =====")
    best_model.eval()
    final_correct, final_total = 0, 0
    all_preds = []
    all_true = []
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            outputs = best_model(features)
            _, predicted = torch.max(outputs, 1)
            final_total += labels.size(0)
            final_correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_true.extend(labels.cpu().numpy())
    
    final_acc = final_correct / final_total
    print(f"\n✅ 最优模型最终测试准确率：{final_acc:.4f}")
    plot_confusion_matrix(all_true, all_preds, CM_SAVE_PATH)
    
    print("\n📢 所有可视化结果已保存：")
    print(f"   - 交互式网络拓扑：{NETWORK_VIS_PATH}（若生成失败可忽略）")
    print(f"   - 权重分布：{WEIGHT_VIS_PATH}")
    print(f"   - 混淆矩阵：{CM_SAVE_PATH}")
    print(f"   - 激活值分布：activation_*.png")
    print(f"   - TensorBoard日志：{LOG_SAVE_PATH}")
    print(f"\n启动TensorBoard命令：tensorboard --logdir={LOG_SAVE_PATH}")
    print(f"TensorBoard访问地址：http://localhost:6006")