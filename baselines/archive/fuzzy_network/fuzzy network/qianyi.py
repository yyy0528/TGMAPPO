import matplotlib.pyplot as plt

# 示例数据
categories = ['A', 'B', 'C', 'D', 'E']
values = [25, 30, 20, 15, 35]

# 设置颜色（每个柱子不同颜色）
colors = ['#FF5733']

# 创建柱状图，设置宽度和透明度
plt.bar(
    categories,
    values,
    width=0.6,         # 设置柱子宽度（默认值为0.8）
    color=colors,      # 设置柱子颜色
    alpha=0.7          # 设置透明度（0.0为完全透明，1.0为完全不透明）
)

# 添加标题和标签
plt.title('自定义柱状图样式')
plt.xlabel('类别')
plt.ylabel('数值')

# 显示图形
plt.show()