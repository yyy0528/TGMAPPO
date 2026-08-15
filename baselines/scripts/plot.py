

import os
import re
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter  # ⭐ 加入移动中位数滤波

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# =========================
# 平滑函数（和之前代码一致）
# =========================
def smooth_data(x, y, window_size=11):
    """
    使用移动中位数进行平滑
    """
    if window_size % 2 == 0:
        window_size += 1  # 确保为奇数

    y_smoothed = median_filter(y, size=window_size)
    return x[:len(y_smoothed)], y_smoothed


# =========================
# 解析txt文件
# =========================
def parse_txt_file(filepath):
    algorithm_name = os.path.basename(filepath).replace('_training_log.txt', '')

    episodes = []
    delivery_rates = []
    average_hops = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and ',' in line:
                parts = line.split(',')
                if len(parts) >= 3:
                    try:
                        episodes.append(int(parts[0]))
                        delivery_rates.append(float(parts[1]))
                        average_hops.append(float(parts[2]))
                    except:
                        continue

    return {
        'algorithm': algorithm_name,
        'episodes': episodes,
        'delivery_rates': delivery_rates,
        'average_hops': average_hops
    }


# =========================
# 更鲁棒的log解析
# =========================
def parse_log_file_robust(filepath):
    algorithm_name = os.path.basename(filepath).replace('_training_log.log', '').replace('_training_log.txt', '')

    episodes = []
    delivery_rates = []
    average_hops = []

    with open(filepath, 'r', encoding='gbk') as f:
        for line in f:
            if 'Episode' in line:
                parts = line.split('|')
                if len(parts) >= 3:
                    try:
                        episode_match = re.search(r'Episode\s+(\d+)', parts[0])
                        if not episode_match:
                            continue
                        episode = int(episode_match.group(1))

                        rate_match = re.search(r'([\d.]+)', parts[1])
                        hops_match = re.search(r'([\d.]+)', parts[2])

                        if rate_match and hops_match:
                            episodes.append(episode)
                            delivery_rates.append(float(rate_match.group(1)))
                            average_hops.append(float(hops_match.group(1)))
                    except:
                        continue

    return {
        'algorithm': algorithm_name,
        'episodes': episodes,
        'delivery_rates': delivery_rates,
        'average_hops': average_hops
    }


# =========================================================
# ⭐ 加入平滑后的总对比绘图函数
# =========================================================
def plot_comparison_all_algorithms(data_list, y_data_type='delivery_rates',
                                   output_path=None, window_size=51):
    """
    绘制所有算法的平滑对比图
    """
    for i, data in enumerate(data_list):
        print(f"Algorithm: {data['algorithm']}")
        print(f"  Episodes: {len(data['episodes'])}")
        print(f"  Delivery Rates: {len(data['delivery_rates'])}")
        print(f"  Average Hops: {len(data['average_hops'])}")
        if data['episodes'] and data[y_data_type]:
            x = data['episodes']
            y = data[y_data_type]
            if len(x) != len(y):
                print(f"Warning: Length mismatch for {data['algorithm']}. Skipping.")
                continue
            # 继续绘图
                # Ensure we are plotting
            print(f"Plotting {data['algorithm']}...")
    plt.figure(figsize=(14, 7))

    colors = plt.cm.tab10(np.linspace(0, 1, len(data_list)))
    # line_styles = ['-', '--', '-.', ':']

    for i, data in enumerate(data_list):
        if data['episodes'] and data[y_data_type]:
            x = data['episodes']
            y = data[y_data_type]

            if len(x) != len(y):
                continue

            # ⭐ 原始曲线（透明）
            plt.plot(x, y, color=colors[i], alpha=0.2, linewidth=1)

            # ⭐ 平滑曲线
            x_smooth, y_smooth = smooth_data(x, y, window_size)

            plt.plot(x_smooth, y_smooth,
                     label=data['algorithm'],
                     color=colors[i],
                     linewidth=3,
                     # linestyle=line_styles[i % len(line_styles)]
                     linestyle='-'  # ⭐ 固定为实线
                    )

    if y_data_type == 'delivery_rates':
        ylabel = 'Delivery Rate'
        title = 'Delivery Rate Comparison'
        plt.ylim(0, 1.05)
    else:
        ylabel = 'Average Hops'
        title = 'Average Hops Comparison'

    # plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)

    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc='best')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")

    plt.show()


# =========================
# 算法名称映射（文件名中的下划线 → 显示用的连字符）
# =========================
ALGORITHM_DISPLAY_NAME = {
    "TG_MAPPO": "TG-MAPPO",
    "QMIX_ATTN": "QMIX-ATTN",
}


def get_display_name(raw_name):
    return ALGORITHM_DISPLAY_NAME.get(raw_name, raw_name)


# =========================
# 主函数
# =========================
def main():
    # ⭐ 只需修改这一个参数即可切换 moves 数量
    NUM_MOVES = 25

    base_dir = os.path.join(PROJECT_ROOT, "results", f"moves_{NUM_MOVES}", "logs")

    algorithms = [
        ("DQN",       "txt"),
        ("IQL",       "txt"),
        ("MAPPO",     "txt"),
        ("RBMADQ",    "txt"),
        ("QMIX_ATTN", "txt"),
        ("QMIX",      "txt"),
        ("VDN",       "txt"),
        ("TG_MAPPO",  "log"),
    ]

    file_paths = [
        os.path.join(base_dir, f"{name}_training_log.{ext}")
        for name, ext in algorithms
    ]

    data_list = []

    for filepath in file_paths:
        if os.path.exists(filepath):
            print(f"Processing: {os.path.basename(filepath)}")

            if filepath.endswith('.log'):
                data = parse_log_file_robust(filepath)
            else:
                data = parse_txt_file(filepath)

            data['algorithm'] = get_display_name(data['algorithm'])

            print(f"  Algorithm: {data['algorithm']}")
            print(f"  Data points: {len(data['episodes'])}")
            print()

            data_list.append(data)

    if not data_list:
        print("No data found.")
        return

    output_dir = os.path.join(PROJECT_ROOT, "results", f"moves_{NUM_MOVES}", "plots_smooth")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Generating smoothed comparison plots...")

    # ⭐ 交付率平滑图
    plot_comparison_all_algorithms(
        data_list,
        y_data_type='delivery_rates',
        output_path=os.path.join(output_dir, "delivery_rate_comparison_smooth.png"),
        window_size=21 # ⭐ 可以调大更平滑
    )

    # ⭐ 平均跳数平滑图
    plot_comparison_all_algorithms(
        data_list,
        y_data_type='average_hops',
        output_path=os.path.join(output_dir, "average_hops_comparison_smooth.png"),
        window_size=21
    )


if __name__ == "__main__":
    main()