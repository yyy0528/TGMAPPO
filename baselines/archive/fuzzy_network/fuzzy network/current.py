import numpy as np

# np.random.seed(42)

base = 0.3
def current_move(nodes, dt, num_steps):
    # 定义节点数量
    num_nodes = len(nodes)

    dt = dt  # 时间步长
    t_total = num_steps

    k1 = np.random.normal(np.pi/base, (0.1 * np.pi)/base, num_nodes)
    k2 = np.random.normal(np.pi/base, (0.1 * np.pi)/base, num_nodes)
    k3 = np.random.normal((2 * np.pi)/base, (0.2 * np.pi)/base, num_nodes)
    A = np.random.normal(3/base, (0.3 * np.pi)/base, num_nodes)
    B = np.random.normal(1/base, 0.1/base, num_nodes)

    print(A)



    sigma = 1  # 噪声标准差
    k4 = np.random.normal(0, sigma, num_nodes)
    k5 = np.random.normal(0, sigma, num_nodes)
    k6 = np.random.normal(0, sigma, num_nodes)

    # 定义时间步长和总时长
    # dt = 0.1  # 时间步长
    # t_total = 20  # 总模拟时间
    # num_steps = int(t_total / dt)  # 总步数

    # 初始化位置数组
    x = np.zeros((num_nodes, num_steps + 1))
    y = np.zeros((num_nodes, num_steps + 1))
    z = np.zeros((num_nodes, num_steps + 1))

    # 计算每个节点的轨迹
    for i in range(num_nodes):
        for t in range(1, num_steps + 1):

            if t == 1:
                x[i, t - 1] = nodes[i][0]
                y[i, t - 1] = nodes[i][1]
                z[i, t - 1] = nodes[i][2]

            # 计算速度分量
            Vx = A[i] * B[i] * k1[i] * np.sin(k2[i] * x[i, t - 1]) * np.cos(k3[i] * y[i, t - 1]) + k1[i] * A[
                i] * np.cos(2 * k1[i] * t * dt) + np.random.normal(0, sigma)
            Vy = -A[i] * B[i] * np.cos(k2[i] * x[i, t - 1]) * np.sin(k3[i] * y[i, t - 1]) + np.random.normal(0, sigma)
            Vz = np.random.normal(0, sigma)

            # 更新位置
            x[i, t] = x[i, t - 1] + Vx * dt
            y[i, t] = y[i, t - 1] + Vy * dt
            z[i, t] = z[i, t - 1] + Vz * dt






    return [x, y, z]