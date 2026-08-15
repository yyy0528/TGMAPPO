import matplotlib.pyplot as plt
import re
import numpy as np
from PIL import Image
import io
from pathlib import Path


# 读取日志文件，提取交付率和平均跳数数据
def extract_data_from_log(log_file):
    episodes = []
    rates = []
    average_hops = []
    with open(log_file, 'r') as f:
        for line in f:
            # 匹配Episode、交付率、平均跳数
            match = re.search(r'Episode (\d+) \| 交付率: ([\d.]+) \| 平均跳数: ([\d.]+) \|', line)
            if match:
                episode = int(match.group(1))
                rate = float(match.group(2))
                avg_hop = float(match.group(3))
                episodes.append(episode)
                rates.append(rate)
                average_hops.append(avg_hop)
    return episodes, rates, average_hops


# 从现有图像中提取数据点
def extract_data_from_image(image_path):
    # 这是一个简化的实现，实际从图像提取数据可能需要更复杂的处理
    # 这里我们假设图像是使用matplotlib保存的，可以通过重新绘制的方式获取数据
    # 更精确的实现可能需要使用计算机视觉技术识别曲线

    # 读取图像
    img = Image.open(image_path)

    # 创建一个新的图形，使用与原图相同的尺寸
    fig, ax = plt.subplots(figsize=(img.width / 100, img.height / 100), dpi=100)

    # 显示图像作为背景
    ax.imshow(img, extent=[0, img.width, 0, img.height], aspect='auto', alpha=0.5)

    # 获取坐标轴范围（假设原图使用了合理的坐标轴范围）
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # 注意：实际应用中，这里需要更复杂的逻辑来检测和提取现有曲线
    # 这个示例中，我们仅返回坐标轴范围作为参考

    plt.close(fig)
    return xlim, ylim


# 在现有rate图像上添加新曲线
def add_curve_to_rate_plot(existing_image_path, new_episodes, new_rates, output_path):
    # 提取现有图像的坐标轴范围
    xlim, ylim = extract_data_from_image(existing_image_path)

    # 读取现有图像
    existing_img = Image.open(existing_image_path)

    # 创建新图形
    fig, ax = plt.subplots(figsize=(existing_img.width / 100, existing_img.height / 100), dpi=100)

    # 显示现有图像
    ax.imshow(existing_img, extent=[xlim[0], xlim[1], ylim[0], ylim[1]], aspect='auto')

    # 添加新曲线
    ax.plot(new_episodes, new_rates, 'g-', label='dyn_reward_gnn_mappo', linewidth=2)

    # 设置坐标轴范围与原图一致
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # 添加图例（如果原图没有图例，可能需要调整位置）
    ax.legend()

    # 保存新图像
    plt.savefig(output_path, bbox_inches='tight', dpi=100)
    plt.close()


# 在现有average_hops图像上添加新曲线
def add_curve_to_hops_plot(existing_image_path, new_episodes, new_hops, output_path):
    # 提取现有图像的坐标轴范围
    xlim, ylim = extract_data_from_image(existing_image_path)

    # 读取现有图像
    existing_img = Image.open(existing_image_path)

    # 创建新图形
    fig, ax = plt.subplots(figsize=(existing_img.width / 100, existing_img.height / 100), dpi=100)

    # 显示现有图像
    ax.imshow(existing_img, extent=[xlim[0], xlim[1], ylim[0], ylim[1]], aspect='auto')

    # 添加新曲线
    ax.plot(new_episodes, new_hops, 'g-', label='dyn_reward_gnn_mappo', linewidth=2)

    # 设置坐标轴范围与原图一致
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # 添加图例
    ax.legend()

    # 保存新图像
    plt.savefig(output_path, bbox_inches='tight', dpi=100)
    plt.close()


if __name__ == "__main__":
    # 日志文件路径
    log_file = Path(__file__).resolve().parents[1] / "outputs" / "training_logs" / "log_20250830_213802.log"

    # 现有图像路径（请替换为实际的图像路径）
    rate_image_path = r"D:\programs\20240607buoy\20240607buoy\buoy_network\交付率对比图1.png"  # 原rate变化曲线图像
    hops_image_path = r"D:\programs\20240607buoy\20240607buoy\buoy_network\平均跳数对比图1.png" # 原average_hops变化曲线图像

    # 输出图像路径
    new_rate_image_path = 'rate_curve_with_new.png'
    new_hops_image_path = 'average_hops_curve_with_new.png'

    # 从日志提取新数据
    episodes_new, rates_new, average_hops_new = extract_data_from_log(log_file)

    # 在现有图像上添加新曲线
    add_curve_to_rate_plot(rate_image_path, episodes_new, rates_new, new_rate_image_path)
    add_curve_to_hops_plot(hops_image_path, episodes_new, average_hops_new, new_hops_image_path)

    print(f"已在原有图像上添加新曲线，并保存为：")
    print(f"{new_rate_image_path}")
    print(f"{new_hops_image_path}")
