import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 参数设置
A = 1.2  # B(t)公式中的常数项，这个其实是洋流的作用范围
c = 0.12  # 波速
k = 2 * np.pi / 7.5  # 波数
omega = 0.4  # 角频率
epsilon = 0.3  # 振幅参数


# 定义B(t)函数
def B(t):
    return A + epsilon * np.cos(omega * t)


# 定义psi(x, y, t)函数
def psi(x, y, t):
    bt = B(t)
    argument = k * (x - c * t)
    numerator = y - bt * np.sin(argument)
    denominator = np.sqrt(1 + (k ** 2) * (bt ** 2) * (np.cos(argument) ** 2))
    return -np.tanh(numerator / denominator)


# 计算x的导数(dot_x = -∂ψ/∂y)
def dxdt(x, y, t):
    # 使用数值微分计算偏导数
    h = 1e-6  # 微小增量
    return -(psi(x, y + h, t) - psi(x, y - h, t)) / (2 * h)


# 计算y的导数(dot_y = ∂ψ/∂x)
def dydt(x, y, t):
    # 使用数值微分计算偏导数
    h = 1e-6  # 微小增量
    return (psi(x + h, y, t) - psi(x - h, y, t)) / (2 * h)


# 使用欧拉方法进行数值积分
def euler_step(x, y, t, dt):
    x_new = x + dxdt(x, y, t) * dt
    y_new = y + dydt(x, y, t) * dt
    return x_new, y_new


# 模拟参数
t_start = 0
t_end = 8
dt = 0.1
num_steps = int((t_end - t_start) / dt)

print(num_steps)

# 初始位置（9个节点，均匀分布在3x3网格上）
nodes = []

nodes = [[x, y] for x in range(3) for y in range(3)]
grid_size = 4  # 3x3网格
spacing = 3.0  # 节点间距
# for i in range(grid_size):
#     for j in range(grid_size):
#         # 计算网格位置，中心在原点附近
#         x = (i - (grid_size - 1) / 2) * spacing
#         y = (j - (grid_size - 1) / 2) * spacing
#         nodes.append(np.array([x, y]))

num_nodes = len(nodes)
print(f"模拟节点数量: {num_nodes}")

# 存储轨迹
trajectories = [np.zeros((num_steps, 2)) for _ in range(num_nodes)]
for i in range(num_nodes):
    trajectories[i][0] = nodes[i]

# 执行模拟
for step in range(1, num_steps):
    t = t_start + step * dt
    for i in range(num_nodes):
        x_prev, y_prev = trajectories[i][step - 1]
        x_new, y_new = euler_step(x_prev, y_prev, t, dt)
        trajectories[i][step] = np.array([x_new, y_new])



# 可视化结果
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('9个节点的运动轨迹')
ax.grid(True)

# 设置坐标轴范围（根据模拟结果调整）
all_x = np.concatenate([traj[:, 0] for traj in trajectories])
all_y = np.concatenate([traj[:, 1] for traj in trajectories])
x_range = max(np.max(all_x) - np.min(all_x), 15)  # 增大范围以更好显示9个节点
y_range = max(np.max(all_y) - np.min(all_y), 15)
ax.set_xlim(np.mean(all_x) - x_range / 2, np.mean(all_x) + x_range / 2)
ax.set_ylim(np.mean(all_y) - y_range / 2, np.mean(all_y) + y_range / 2)

# 绘制轨迹 - 使用不同颜色区分节点
colors = plt.cm.Set3(np.linspace(0, 1, num_nodes))  # 使用颜色映射生成9种不同颜色
lines = []
points = []
for i in range(num_nodes):
    line, = ax.plot([], [], color=colors[i], alpha=0.5)
    point = ax.scatter([], [], color=colors[i], s=50, label=f'节点 {i + 1}')
    lines.append(line)
    points.append(point)

ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # 将图例放在右侧


# 动画更新函数
def update(frame):
    for i in range(num_nodes):
        # 更新轨迹线
        lines[i].set_data(trajectories[i][:frame, 0], trajectories[i][:frame, 1])
        # 更新节点位置
        points[i].set_offsets([[trajectories[i][frame, 0], trajectories[i][frame, 1]]])
    return lines + points


# 创建动画
ani = FuncAnimation(fig, update, frames=num_steps, interval=50, blit=False)

plt.tight_layout()
plt.show()
