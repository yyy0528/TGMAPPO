import matplotlib.pyplot as plt

import matplotlib as mpl

plt.figure(figsize=(6, 5), constrained_layout=True)
plt.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['mathtext.fontset'] = 'stix'


# 示例数据
categories = ['27->64', 'random->64', '64->64']
values = [7892, 590, 5435]
# values = [0.580 ,0.687, 0.869, 0.859, 0.874]

# 设置颜色（每个柱子不同颜色）
colors = ['r', 'b','b','b','b']

# 创建柱状图，设置宽度和透明度
plt.bar(
    categories,
    values,
    width=0.5,         # 设置柱子宽度（默认值为0.8）
    color= colors,      # 设置柱子颜色
    alpha=0.7        # 设置透明度（0.0为完全透明，1.0为完全不透明）
)

# 添加标题和标签
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
# plt.xlabel('scenarios', fontsize=20)
plt.ylabel('return', fontsize=20)
# plt.ylabel('Delay/s', fontsize=20)

# from matplotlib.patches import Patch
# legend_elements = [Patch(facecolor='r', edgecolor='r', alpha=0.7, label='Training scenario'),
#                    Patch(facecolor='b', edgecolor='b', alpha=0.7, label='Transfer scenario')]

# plt.legend(handles=legend_elements, fontsize=16, loc='upper right')


# 显示图形
plt.show()
# plt.savefig('Fig11.1.png', dpi=200)