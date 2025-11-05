import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.ndimage import median_filter  # 用于计算移动中位数

# 设置中文字体，确保中文正常显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


def parse_log_file(file_path):
    """解析日志文件，提取episode、交付率、平均跳数"""
    episodes = []
    rates = []
    average_hops = []

    # 日志行的正则匹配模式
    pattern = r"Episode (\d+) \| 交付率: (\d+\.\d+) \| 平均跳数: (\d+\.\d+) \| 平均时延: (\d+\.\d+) \|"

    # 尝试多种编码格式（解决解码错误）
    encodings = ['utf-8', 'gbk', 'gb2312', 'ansi']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                for line in f:
                    match = re.search(pattern, line)
                    if match:
                        episode = int(match.group(1))
                        rate = float(match.group(2))
                        hop = float(match.group(3))

                        episodes.append(episode)
                        rates.append(rate)
                        average_hops.append(hop)
            # 如果成功解析，跳出编码尝试循环
            print(f"使用编码 {encoding} 成功解析文件: {file_path}")
            break
        except UnicodeDecodeError:
            continue
    else:
        # 所有编码都尝试失败
        raise ValueError(f"无法解析文件 {file_path}，尝试过的编码: {encodings}")

    return episodes, rates, average_hops

def smooth_data(x, y, window_size=11):
    """
    用移动中位数平滑数据
    :param x: 横坐标数据（episode）
    :param y: 纵坐标数据（rate/hops）
    :param window_size: 滑动窗口大小（需为奇数，越大越平滑）
    :return: 平滑后的x, y
    """
    # 确保窗口大小为奇数
    if window_size % 2 == 0:
        window_size += 1
    # 计算移动中位数
    y_smoothed = median_filter(y, size=window_size)
    # 保持x与y长度一致（仅取有效部分）
    return x[:len(y_smoothed)], y_smoothed

def plot_comparison(data1, data2, label1, label2, y_label, title, save_path, window_size=11):
    """绘制两个数据集的对比图（带数据平滑）"""
    episodes1, values1 = data1
    episodes2, values2 = data2

    # 对数据进行平滑处理![](交付率对比图1.png)
    episodes1_smooth, values1_smooth = smooth_data(episodes1, values1, window_size)
    episodes2_smooth, values2_smooth = smooth_data(episodes2, values2, window_size)

    plt.figure(figsize=(6, 4))

    # 绘制原始数据（可选：透明浅色填充，展示波动范围）
    plt.plot(episodes1, values1, color='red', alpha=0.2, linewidth=1)
    plt.plot(episodes2, values2, color='blue', alpha=0.2, linewidth=1)

    # 绘制平滑后的数据曲线
    plt.plot(episodes1_smooth, values1_smooth, label=label1, linewidth=2, color='red')
    plt.plot(episodes2_smooth, values2_smooth, label=label2, linewidth=2, color='blue')

    # 添加标题和坐标轴标签
    plt.title(title, fontsize=14)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel(y_label, fontsize=12)

    # 添加网格和图例
    plt.grid(alpha=0.3)
    plt.legend(fontsize=12)

    # 保存图片
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"对比图已保存至: {save_path}")


if __name__ == "__main__":
    # 两个日志文件的路径（请根据实际路径修改）
    log_path1 = r"D:\programs\20251012\20250829buoy\20240607buoy\buoy_network\training_logs\log_20251019_231801.log"
    log_path2 = r"D:\programs\20251012\20250829buoy\20240607buoy\buoy_network\training_logs\log_20251019_211749.log"

    # 解析日志文件
    episodes1, rates1, hops1 = parse_log_file(log_path1)
    episodes2, rates2, hops2 = parse_log_file(log_path2)

    # 确保两个日志的episode范围一致（取最小长度）
    min_len = min(len(episodes1), len(episodes2))
    episodes1 = episodes1[:min_len]
    rates1 = rates1[:min_len]
    hops1 = hops1[:min_len]

    episodes2 = episodes2[:min_len]
    rates2 = rates2[:min_len]
    hops2 = hops2[:min_len]

    # 数据集标签（可根据实际实验名称修改）
    label1 = "dqn"
    label2 = "mappo_gnn"

    # 绘制交付率对比图（调整 window_size 控制平滑程度，如11、21等）
    rate_data1 = (episodes1, rates1)
    rate_data2 = (episodes2, rates2)
    plot_comparison(
        rate_data1, rate_data2,
        label1, label2,
        y_label="rate",
        title="compare_rate",
        save_path="交付率对比图2.png",
        window_size=21  # 窗口越大，曲线越平滑
    )

    # 绘制平均跳数对比图
    hop_data1 = (episodes1, hops1)
    hop_data2 = (episodes2, hops2)
    plot_comparison(
        hop_data1, hop_data2,
        label1, label2,
        y_label="average_hops",
        title="compare_average_hops",
        save_path="平均跳数对比图2.png",
        window_size=21
    )