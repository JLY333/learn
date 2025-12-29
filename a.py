import pandas as pd
import os
import re

# ===================== 核心配置（仅需改这1个路径） =====================
# 你的CSV文件所在目录（包含ib600_2.csv等文件的文件夹）
csv_dir = r"C:\Users\ASUS\Desktop\故障诊断\JNU-Bearing-Dataset" 
output_dir = csv_dir
os.makedirs(output_dir, exist_ok=True)

# ===================== 定义映射关系 =====================
fault_label_map = {
    "n": 0,    # 正常
    "ib": 1,   # 内圈故障
    "tb": 2,   # 滚动体故障
    "ob": 3    # 外圈故障
}
target_rpms = [600, 800, 1000]  # 目标转速
sampling_freq = 50000           # 江南大学数据集采样频率50kHz

# ===================== 初始化转速分组 =====================
rpm_groups = {rpm: [] for rpm in target_rpms}

# ===================== 正则匹配工具 =====================
rpm_pattern = re.compile(r"600|800|1000")       # 匹配转速
fault_pattern = re.compile(r"n|ib|ob|tb", re.IGNORECASE)  # 匹配故障类型

# ===================== 遍历并处理所有CSV文件 =====================
print("===== 开始处理CSV文件 =====")
for file in os.listdir(csv_dir):
    if not file.endswith(".csv"):
        continue  # 只处理CSV文件
    
    file_path = os.path.join(csv_dir, file)
    try:
        # 读取CSV：先尝试带列名，若失败则无列名（header=None）
        try:
            df = pd.read_csv(file_path)
        except:
            df = pd.read_csv(file_path, header=None)
        
        # 确定振动信号的列（优先取第一列，兼容无列名/列名错误）
        signal_col = df.columns[0]  # 无论列名是什么，取第一列作为振动信号
        df.rename(columns={signal_col: "振动信号"}, inplace=True)  # 统一列名为“振动信号”
        
        # 提取转速（从文件名）
        rpm_match = rpm_pattern.search(file)
        if not rpm_match:
            print(f"⚠️ 跳过 {file}：未识别到转速（600/800/1000）")
            continue
        rpm = int(rpm_match.group())
        
        # 提取故障类型（从文件名）
        fault_match = fault_pattern.search(file.lower())
        if not fault_match:
            print(f"⚠️ 跳过 {file}：未识别到故障类型（n/ib/ob/tb）")
            continue
        fault = fault_match.group().lower()
        
        # 添加元数据列
        df["故障标签"] = fault_label_map[fault]  # 故障标签
        df["转速(rpm)"] = rpm                  # 转速
        df["采样频率(Hz)"] = sampling_freq     # 采样频率
        df["源文件"] = file                     # 源文件名
        
        # 只保留需要的列（避免冗余）
        df = df[["振动信号", "故障标签", "采样频率(Hz)", "转速(rpm)", "源文件"]]
        
        # 加入对应转速分组
        if rpm in rpm_groups:
            rpm_groups[rpm].append(df)
        print(f"✅ 处理成功：{file} → {rpm}rpm，{fault}故障")
    
    except Exception as e:
        print(f"❌ 处理失败 {file}：{str(e)[:50]}...")  # 简化错误信息

# ===================== 按转速合并并保存文件 =====================
print("\n===== 按转速合并数据 =====")
for rpm, df_list in rpm_groups.items():
    if not df_list:
        print(f"⚠️ 无 {rpm}rpm 的有效数据，跳过")
        continue
    
    # 合并当前转速的所有数据
    combined_df = pd.concat(df_list, ignore_index=True)
    
    # 保存文件
    output_path = os.path.join(output_dir, f"JNU_轴承故障_{rpm}rpm.csv")
    combined_df.to_csv(output_path, index=False, encoding="utf-8")
    
    # 输出统计信息
    print(f"\n📊 {rpm}rpm 整合结果：")
    print(f"   总数据行数：{len(combined_df)}")
    print(f"   故障类型分布：")
    for fault, label in fault_label_map.items():
        count = len(combined_df[combined_df["故障标签"] == label])
        if count > 0:
            print(f"     - {fault}（标签{label}）：{count} 行")
    print(f"✅ 保存完成：{output_path}")

# ===================== 最终提示 =====================
print("\n🎉 全部处理完成！")
# 列出生成的文件
print("\n生成的文件清单：")
for rpm in target_rpms:
    file = os.path.join(output_dir, f"JNU_轴承故障_{rpm}rpm.csv")
    if os.path.exists(file):
        print(f"   ✅ {file}")
    else:
        print(f"   ❌ {rpm}rpm 文件未生成（无数据）")

