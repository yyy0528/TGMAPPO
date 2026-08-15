

import matplotlib.pyplot as plt
import numpy as np
import os
import re

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def extract_data_from_file(file_path):
    """
    从文件中提取时间延迟数据（第三列，或从日志中提取）
    """
    delays = []
    if file_path.endswith(".txt"):
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.split(',')
                if len(parts) >= 3:
                    delays.append(float(parts[2]))
    elif file_path.endswith(".log"):
        with open(file_path, 'r') as file:
            for line in file:
                # 匹配 "| ƽ��ʱ��: 13.78 |" 中的数字
                # match = re.search(r'\|\s*ƽ��ʱ��:\s*(\d+\.\d+)\s*\|', line)
                # match = re.search(r'\|\s*(\d+\.\d+)\s*\|\s*temp_reward', line)
                match = re.search(r'(\d+\.\d+)\s*\|\s*temp_reward', line)
                print(match)
                if match:
                    delays.append(float(match.group(1)))
    return delays

def categorize_delays(delays):
    """
    将延迟分为四个区间，返回计数列表
    """
    categories = [0, 0, 0, 0]
    for d in delays:
        if d <= 17:
            categories[0] += 1
        elif d <= 18:
            categories[1] += 1
        elif d <= 20:
            categories[2] += 1
        else:
            categories[3] += 1
    return categories
def plot_stacked_bar_chart(data_dict, algorithm_order):
    labels = [name for name in algorithm_order if name in data_dict]
    percentages = np.array([data_dict[name] for name in labels])

    colors = ['#7fd7a7', '#ffffbf', '#ffbfbf', '#cceef8']
    categories = ['0-17 ms', '17-18 ms', '18-20 ms', '>20 ms']

    x = np.arange(len(labels))
    bar_width = 0.6
    bottom = np.zeros(len(labels))

    fig, ax = plt.subplots(figsize=(12, 6))

    for i in range(4):
        bars = ax.bar(
            x,
            percentages[:, i],
            bar_width,
            bottom=bottom,
            color=colors[i],
            label=categories[i],
            edgecolor='none',
            zorder = 3  # Ensure bars are drawn above the grid lines
        )

        # ====== 在每个堆叠块上添加百分比文字 ======
        for j in range(len(labels)):
            height = percentages[j, i]
            if height > 0:  # 只标注非0的
                ax.text(
                    x[j],
                    bottom[j] + height / 2,     # 文字在当前块中间
                    f'{height:.1f}%',           # 保留1位小数
                    ha='center',
                    va='center',
                    fontsize=15,
                    color='black'
                )

        bottom += percentages[:, i]

    # ax.set_xlabel('Algorithms', fontsize=12)
    # ax.set_ylabel('Percentage (%)', fontsize=12)
    # ax.set_title('Time Delay Distribution by Algorithm', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylim(0, 100)
    # Format y-axis labels as percentages
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_yticklabels([f'{i}%' for i in np.arange(0, 101, 20)])
    ax.legend(loc='upper right')
    # Remove the black lines around the plot
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Hide grid lines on both axes
    ax.grid(False)  # This will hide all grid lines
    # Optional: Hide the ticks on the x and y axes as well
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
    # Enable grid lines for y-axis
    # ax.grid(axis='y', linestyle='--', linewidth=0.5)  # Dashed grid lines along y-axis
    ax.grid(axis='y', linestyle='-', linewidth=0.5, color='#ebebeb', zorder=0)  # Lighter gray color for grid lines
    plt.tight_layout()
    out_dir = os.path.join(PROJECT_ROOT, "results", "time_delay")
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f'time_delay_stacked_bar_{NUM_MOVES}.png'), dpi=300, bbox_inches='tight')
    plt.show()

# ⭐ 只需修改这一个参数即可切换 moves 数量
NUM_MOVES = 25

base_dir = os.path.join(PROJECT_ROOT, "results", f"moves_{NUM_MOVES}", "logs")

files = [
    os.path.join(base_dir, "dqn_training_log.txt"),
    os.path.join(base_dir, "iql_training_log.txt"),
    os.path.join(base_dir, "mappo_training_log.txt"),
    os.path.join(base_dir, "RBMADQ_training_log.txt"),
    os.path.join(base_dir, "qmix_Attn_training_log.txt"),
    os.path.join(base_dir, "qmix_training_log.txt"),
    os.path.join(base_dir, "vdn_training_log.txt"),
    os.path.join(base_dir, "TG_Mappo_training_log.log"),
]

# ========== 您指定的算法名称（顺序与文件一一对应） ==========
algorithm_names = ["DQN", " IQL", "MAPPO", "RBMADQ", "QMIX-ATTN", "QMIX", "VDN", "TG-MAPPO"]

# ========== 数据处理 ==========
raw_counts = {}          # 存储每个算法的四个区间计数
for file, name in zip(files, algorithm_names):
    delays = extract_data_from_file(file)
    categories = categorize_delays(delays)
    raw_counts[name] = categories

# 将计数转换为百分比
percent_data = {}
for name, counts in raw_counts.items():
    total = sum(counts)
    if total > 0:
        percent_data[name] = [c / total * 100 for c in counts]
    else:
        percent_data[name] = [0, 0, 0, 0]

# 绘制图表，横轴顺序即为 algorithm_names
plot_stacked_bar_chart(percent_data, algorithm_names)