import numpy as np


t_start = 0
t_end = 50
dt = 0.1



A = 1.2  # B(t)公式中的常数项，这个其实是洋流的作用范围
c = 0.12  # 波速
k = 2.5 * np.pi / 7.5  # 波数
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
def euler_step(x, y, z, t, dt):
    x_new = x + dxdt(x, y, t) * dt + np.random.normal(0, 0.01)
    y_new = y + dydt(x, y, t) * dt + np.random.normal(0, 0.01)
    z_new = z + np.random.normal(0, 0.01)
    # x_new = x + dxdt(x, y, t) * dt
    # y_new = y + dydt(x, y, t) * dt
    # z_new = z
    return x_new, y_new, z_new



def current_move(nodes, dt, num_steps):
    num_nodes = len(nodes)
    trajectories = [np.zeros((num_steps, 3)) for _ in range(num_nodes)]
    for i in range(num_nodes):
        trajectories[i][0] = nodes[i]

    # 执行模拟
    for step in range(1, num_steps):
        t = t_start + step * dt
        for i in range(num_nodes):
            x_prev, y_prev, z_prev = trajectories[i][step - 1]
            x_new, y_new, z_new = euler_step(x_prev, y_prev, z_prev, t, dt)
            # z_new = Vz = np.random.normal(0, sigma)
            trajectories[i][step] = np.array([x_new, y_new, z_new])

    # print(trajectories[1])
    return trajectories

# def current_move(nodes, dt, num_steps):
#     num_nodes = len(nodes)
#     trajectories = [np.zeros((num_nodes, 3)) for _ in range(num_steps)]
#     for i in range(num_nodes):
#         trajectories[0][i] = nodes[i]
#
#     # 执行模拟
#     for step in range(1, num_steps):
#         t = t_start + step * dt
#         for i in range(num_nodes):
#             x_prev, y_prev, z_prev = trajectories[step - 1][i]
#             x_new, y_new, z_new = euler_step(x_prev, y_prev, z_prev, t, dt)
#             # z_new = Vz = np.random.normal(0, sigma)
#             trajectories[step][i] = np.array([x_new, y_new, z_new])
#
#
#
#     xxx = trajectories
#
#     return trajectories
