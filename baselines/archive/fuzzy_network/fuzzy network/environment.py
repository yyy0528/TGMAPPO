import math
import statistics

import numpy as np
from node import Node, DataPacket, Network
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# random.seed(123)


class Environment:
    def __init__(self, nodes):
        self.n_node = len(nodes)
        self.sink_position = None  # sink节点坐标
        self.sink_id = None
        self.nodes_info = nodes  # 创建节点的信息
        self.nodes = None  # 节点对象
        self.network = None
        self.lambda_val = 2
        self.packets = []
        self.packet_id = 0
        self.packet_generation_number = []
        self.packet_length = 100  # 单位 bit

        self.last_state = None


    def build_env(self):

        self.network = Network(self.nodes_info)

        self.nodes = self.network.get_nodes()
        self.sink_id = len(self.nodes) - 1
        sink_node = self.nodes[self.sink_id]

        sink_node.if_sink = True
        sink_node.energy = 9999
        info = self.get_net_info()
        # self.packet_generation_test(5)
        self.packet_generation()

        self.last_state = self.get_state()

        return info

    def reset(self):

        self.network.reset()
        self.packets = []
        self.packet_id = 0
        # self.packet_generation_test(5)
        self.packet_generation()
        obs_all = self.get_obs_all()
        self.last_state = self.get_state()
        return obs_all

    def reset_test(self, packet_number):
        self.network.reset()
        self.packets = []
        self.packet_id = 0
        self.packet_generation_test(packet_number)
        # self.packet_generation()
        obs_all = self.get_obs_all()
        return obs_all

    def step(self, forward_id):
        done = np.zeros(self.n_node)
        sink_node = self.nodes[self.sink_id]

        energy = []
        r_sink = 0
        for n_id in range(self.n_node-1):


            sender = self.nodes[n_id]

            # 当缓冲区没有数据包了，那就不转发，这个应该没有什么问题
            if sender.packet.size() == 0:
                continue

            # 没邻居了也不转发
            if forward_id[n_id] == -1:
                continue
            neighbor = sender.get_neighbor()

            if self.sink_id in neighbor:
                receiver_id = self.sink_id
            else:
                receiver_id = neighbor[forward_id[n_id]]

            receiver = self.nodes[receiver_id]
            packet_id = sender.send(self.packet_length, receiver.position)

            receiver.receive(packet_id, self.packet_length)
            self.packets[packet_id].transmit(receiver_id)


        # 前面是环境交互的部分，下面开始计算奖励

        for n_id in range(self.n_node - 1):
            energy.append(self.nodes[n_id].energy/100)

        v_energy = statistics.pvariance(energy)
        r_e = - v_energy

        dis = 0
        in_sink = 0
        for packet in self.packets:

            if packet.if_sink:
                in_sink += 10

            cur_node = packet.current_node
            pos = self.nodes[cur_node].position
            dis -= math.sqrt(math.pow(pos[0] - sink_node.position[0], 2) + math.pow(pos[1] - sink_node.position[1], 2) + math.pow(pos[2] - sink_node.position[2], 2))

        # reward = np.full(self.n_node, (dis + r_e)/self.packet_id)
        reward = np.full(self.n_node, (dis/100 + in_sink)/self.packet_id + r_e/self.n_node)

        self.move()
        return reward, done


    def step_test(self, forward_id):
        done = np.zeros(self.n_node)
        sink_node = self.nodes[self.sink_id]

        energy = []
        r_sink = 0
        for n_id in range(self.n_node - 1):

            sender = self.nodes[n_id]

            # 当缓冲区没有数据包了，那就不转发，这个应该没有什么问题
            if sender.packet.size() == 0:
                continue

            # 没邻居了也不转发
            if forward_id[n_id] == -1:
                continue
            neighbor = sender.get_neighbor()

            if self.sink_id in neighbor:
                receiver_id = self.sink_id
            else:
                receiver_id = neighbor[forward_id[n_id]]

            receiver = self.nodes[receiver_id]
            packet_id = sender.send(self.packet_length, receiver.position)

            receiver.receive(packet_id, self.packet_length)
            self.packets[packet_id].transmit(receiver_id)

        # 前面是环境交互的部分，下面开始计算奖励

        for n_id in range(self.n_node - 1):
            energy.append(self.nodes[n_id].energy)

        v_energy = statistics.pvariance(energy)
        r_e = - v_energy

        dis = 0
        in_sink = 0
        for packet in self.packets:
            packet.dy += 1

            if packet.if_sink:
                packet.dy += 1
                in_sink += 10

            cur_node = packet.current_node
            pos = self.nodes[cur_node].position
            dis -= math.sqrt(
                math.pow(pos[0] - sink_node.position[0], 2) + math.pow(pos[1] - sink_node.position[1], 2) + math.pow(
                    pos[2] - sink_node.position[2], 2))

        # reward = np.full(self.n_node, (dis + r_e)/self.packet_id)
        reward = np.full(self.n_node, (dis / 100 + in_sink) / self.packet_id + r_e / self.n_node)

        self.move()

        return reward, done, v_energy


    # 自己的位置，缓冲区数据包的数量，邻居的个数，邻居距离sink节点的平均位置，
    def get_obs(self, node_id):
        node = self.nodes[node_id]
        obs_all = []
        for candidate_id in node.neighbor:
            obs = []
            candidate = self.nodes[candidate_id]

            # sink节点位置
            sink_node = self.nodes[self.sink_id]
            # obs.append(sink_node.position[0])
            # obs.append(sink_node.position[1])
            # obs.append(sink_node.position[2])

            obs.append(sink_node.position[0]/100)
            obs.append(sink_node.position[1]/100)
            obs.append(sink_node.position[2]/100)


            # 节点ID
            obs.append(candidate.node_id)

            # 候选节点缓冲区数据包的数量
            obs.append(candidate.packet.size())
            # 候选节点的位置
            # obs.append(candidate.position[0])
            # obs.append(candidate.position[1])
            # obs.append(candidate.position[2])
            obs.append(candidate.position[0]/100)
            obs.append(candidate.position[1]/100)
            obs.append(candidate.position[2]/100)
            # 候选节点的能量
            obs.append(candidate.energy/100)
            # 候选节点邻居的缓冲区数据包的平均数量和平均位置

            c_neighbor_id = candidate.get_neighbor()
            c_neighbors = []
            if c_neighbor_id == []:
                obs.append(0)
                # 没有邻居了，这个还不能设为0，就很烦
                # obs.append(100)
                # obs.append(100)
                # obs.append(100)
                # obs.append(candidate.position[0])
                # obs.append(candidate.position[1])
                # obs.append(candidate.position[2])

                obs.append(candidate.position[0] / 100)
                obs.append(candidate.position[1] / 100)
                obs.append(candidate.position[2] / 100)

                # obs.append(candidate.energy)

            else:


                for ne_id in c_neighbor_id:
                    c_neighbors.append(self.nodes[ne_id])
                packet_tmp = 0

                x_pos = 0
                y_pos = 0
                z_pos = 0

                energy = 0
                for c_neighbor in c_neighbors:
                    # 数据包
                    packet_tmp += c_neighbor.packet.size()
                    # 位置
                    pos = c_neighbor.position
                    x_pos += pos[0]/100
                    y_pos += pos[1]/100
                    z_pos += pos[2]/100
                    # 候选节点邻居们的平均能量
                    energy += c_neighbor.energy/100

                # 这里有除法操作，所以导致了[NAN]
                obs.append(packet_tmp / len(c_neighbors))


                obs.append(x_pos / len(c_neighbors))
                obs.append(y_pos / len(c_neighbors))
                obs.append(z_pos / len(c_neighbors))

                # obs += obs

                # 标准化是有问题的,这个把一些变化的维度也缩减了
            # obs = self.min_max_normalize(obs)

                # obs.append(energy / len(c_neighbors))

            obs_all.append(obs)
        return obs_all

    def get_obs_all(self):
        obs_all = []
        for node_id in range(self.n_node):
            obs = self.get_obs(node_id)
            obs_all.append(obs)
        return obs_all


    def get_n_obs(self):
        obs = self.get_obs(1)

        return len(obs[0])

    # 后续要考虑要不要加编号
    def get_state(self):

        state = []
        for node in self.nodes:


            # 自己缓冲区数据包的数量
            # obs.append(node.packet.size())
            obs = []
            # sink节点位置
            candidate = node

            # sink节点位置
            sink_node = self.nodes[self.sink_id]
            # obs.append(sink_node.position[0])
            # obs.append(sink_node.position[1])
            # obs.append(sink_node.position[2])

            obs.append(sink_node.position[0] / 100)
            obs.append(sink_node.position[1] / 100)
            obs.append(sink_node.position[2] / 100)

            # 节点ID
            obs.append(candidate.node_id)

            # 候选节点缓冲区数据包的数量
            obs.append(candidate.packet.size())
            # 候选节点的位置
            # obs.append(candidate.position[0])
            # obs.append(candidate.position[1])
            # obs.append(candidate.position[2])
            obs.append(candidate.position[0] / 100)
            obs.append(candidate.position[1] / 100)
            obs.append(candidate.position[2] / 100)
            # 候选节点的能量
            obs.append(candidate.energy / 100)
            # 候选节点邻居的缓冲区数据包的平均数量和平均位置

            c_neighbor_id = candidate.get_neighbor()
            c_neighbors = []
            if c_neighbor_id == []:
                obs.append(0)
                # 没有邻居了，这个还不能设为0，就很烦
                # obs.append(100)
                # obs.append(100)
                # obs.append(100)
                # obs.append(candidate.position[0])
                # obs.append(candidate.position[1])
                # obs.append(candidate.position[2])

                obs.append(candidate.position[0] / 100)
                obs.append(candidate.position[1] / 100)
                obs.append(candidate.position[2] / 100)

                # obs.append(candidate.energy)

            else:

                for ne_id in c_neighbor_id:
                    c_neighbors.append(self.nodes[ne_id])
                packet_tmp = 0

                x_pos = 0
                y_pos = 0
                z_pos = 0

                energy = 0
                for c_neighbor in c_neighbors:
                    # 数据包
                    packet_tmp += c_neighbor.packet.size()
                    # 位置
                    pos = c_neighbor.position
                    x_pos += pos[0] / 100
                    y_pos += pos[1] / 100
                    z_pos += pos[2] / 100
                    # 候选节点邻居们的平均能量
                    energy += c_neighbor.energy/100

                # 这里有除法操作，所以导致了[NAN]
                obs.append(packet_tmp / len(c_neighbors))

                obs.append(x_pos / len(c_neighbors))
                obs.append(y_pos / len(c_neighbors))
                obs.append(z_pos / len(c_neighbors))


                # obs.append(energy / len(c_neighbors))
                # 标准化
            # obs = self.min_max_normalize(obs)

            state.append(obs)

        return state

    # def get_fuzzy_state(self):
    #     state = self.get_state()
    #     num_columns = len(state[0])
    #     averages = []
    #
    #     for col in range(num_columns):
    #         column_sum = 0
    #         for row in state:
    #             column_sum += row[col]
    #         average = column_sum / len(state)
    #         averages.append(average)
    #     return averages

    # 分成8个区域的版本
    # def get_fuzzy_state(self):
    #     state = self.get_state()
    #     region = self.partition_nodes([1, 1, 1])
    #
    #     fuzzy_state = []
    #
    #     for i in range(8):
    #         region_nodes = region[i]
    #         region_state = []
    #         region_averages = []
    #         for node_id in region_nodes:
    #             region_state.append(state[node_id])
    #
    #         if region_state == []:
    #             fuzzy_state += [0.]*10
    #
    #             continue
    #
    #         num_columns = len(region_state[0])
    #
    #         for col in range(num_columns):
    #             column_sum = 0
    #             for row in region_state:
    #                 column_sum += row[col]
    #             average = column_sum / len(region_state)
    #             region_averages.append(average)
    #         fuzzy_state += region_averages
    #
    #     r_state = [fuzzy_state for i in range(self.n_node)]
    #     return r_state

    # 节点数目，移动趋势，能量分布

    def get_fuzzy_state(self):
        state = self.get_state()
        last_state = self.last_state
        dx = 0
        dy = 0
        dz = 0
        last_e = []
        e = []
        for i in range(self.n_node):
            dx += state[i][2] - last_state[i][2]
            dy += state[i][3] - last_state[i][3]
            dz += state[i][4] - last_state[i][4]
            last_e.append(last_state[i][5]/100)
            e.append(state[i][5]/100)

        # v_last_e = statistics.pvariance(last_e)
        v_e = statistics.pvariance(e)
        # de = v_e - v_last_e


        transposed_list = list(zip(*state))
        # transposed_list 会是: [(1, 4, 7), (2, 5, 8), (3, 6, 9)]

        # 2. 计算每一列的平均值
        col_means = [sum(col) / len(col) for col in transposed_list]


        # 测试一下消融效果
        tmp = [self.n_node, dx/100, dy/100, dz/100, v_e] + col_means
        # tmp = col_means
        # tmp = [self.n_node, dx / 100, dy / 100, dz / 100, v_e]
        # tmp = self.min_max_normalize(tmp)




        fuzzy_state = [tmp for i in range(self.n_node)]
        # fuzzy_state = [[0.] for i in range(self.n_node)]

        # self.last_state = state

        return fuzzy_state



    def partition_nodes(self, reference_point):
        """
        将节点按照参考点沿三个坐标轴划分为8个区域，并返回每个区域的节点编号

        参数:
            nodes: 字典，键为节点编号，值为节点的三维坐标列表 [x, y, z]
            reference_point: 列表，参考点的三维坐标 [x, y, z]

        返回:
            字典，键为区域编号(0-7)，值为该区域内的节点编号列表
            区域划分规则:
            0: x≥ref_x, y≥ref_y, z≥ref_z
            1: x<ref_x,  y≥ref_y, z≥ref_z
            2: x≥ref_x, y<ref_y,  z≥ref_z
            3: x<ref_x,  y<ref_y,  z≥ref_z
            4: x≥ref_x, y≥ref_y, z<ref_z
            5: x<ref_x,  y≥ref_y, z<ref_z
            6: x≥ref_x, y<ref_y,  z<ref_z
            7: x<ref_x,  y<ref_y,  z<ref_z
        """
        # 初始化8个区域
        regions = {i: [] for i in range(8)}
        # regions = []

        # 提取参考点的三个坐标
        ref_x, ref_y, ref_z = reference_point

        # 遍历所有节点，判断每个节点属于哪个区域
        for node in self.nodes:
            x, y, z = node.position

            # 确定x方向位置 (0表示≥参考点，1表示<参考点)
            x_pos = 0 if x >= ref_x else 1
            # 确定y方向位置
            y_pos = 0 if y >= ref_y else 1
            # 确定z方向位置
            z_pos = 0 if z >= ref_z else 1

            # 计算区域编号 (0-7)
            region_id = x_pos + 2 * y_pos + 4 * z_pos

            # 将节点添加到对应的区域
            regions[region_id].append(node.node_id)

        return regions


    def get_remaining_energy(self):

        energy = np.zeros(len(self.nodes))
        i = 0
        for node in self.nodes:
            energy[i] = node.energy
            i += 1

        mean = np.mean(energy)
        std_dev = np.std(energy)
        return mean, std_dev


    def get_net_info(self):
        info = []
        for node in self.nodes:
            info.append(node.get_info())

        return info

    def get_net_info_id(self, net_id):
        net = self.network[net_id]
        info = net.get_info()

        return info



    def get_packet(self, node_id):

        return self.nodes[node_id].packet

    def packet_generation(self):

        if self.packet_generation_number == []:
            for node in self.nodes:
                if node.if_sink:
                    continue

            # 这里可以用一个列表，让随机种子只作用一次
                random_number = np.random.poisson(lam=self.lambda_val)

                node.n_packet += random_number
                self.packet_generation_number.append(random_number)

                for i in range(random_number):
                    packet = DataPacket(self.packet_id, node.node_id, self.sink_id)
                    node.packet.enqueue(self.packet_id)
                    self.packets.append(packet)
                    self.packet_id += 1
        else:
            for node in self.nodes:
                node.n_packet += self.packet_generation_number[node.node_id]
                self.packet_generation_number.append(self.packet_generation_number[node.node_id])

                for i in range(self.packet_generation_number[node.node_id]):
                    packet = DataPacket(self.packet_id, node.node_id, self.sink_id)
                    node.packet.enqueue(self.packet_id)
                    self.packets.append(packet)
                    self.packet_id += 1

    def packet_generation_test(self, number):
        for node in self.nodes:
            if node.if_sink:
                continue

            node.n_packet += number
            for i in range(number):
                packet = DataPacket(self.packet_id, node.node_id, self.sink_id)
                node.packet.enqueue(self.packet_id)
                self.packets.append(packet)
                self.packet_id += 1

    def packet_generation_one(self, number):
        node = self.nodes[0]
        node.n_packet += 1
        node.packet.enqueue(self.packet_id)
        for i in range(number):
            packet = DataPacket(self.packet_id, node.node_id, self.sink_id)
            self.packets.append(packet)
            self.packet_id += 1

    def move(self):
        self.network.move()

    def min_max_normalize(self, lst):
        x_min = min(lst)
        x_max = max(lst)
        # 处理极端情况：若所有元素相同（max=min），避免除以0，直接返回全0列表
        if x_max == x_min:
            return [0.0 for _ in lst]
        return [(x - x_min) / (x_max - x_min) for x in lst]

    def render(self):
        plt.rcParams['font.family'] = 'Times New Roman'  # Windows/Linux/macOS 通用名称
        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['font.size'] = 14
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection='3d')

        # 设置坐标轴范围和标签
        ax.set_xlim([0, 1000])  # X 范围
        ax.set_ylim([0, 1000])  # Y 范围
        ax.set_zlim([-1000, 0])  # Z 范围（深度从 -100 到 0）
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Depth (m)')  # 深度标签更新
        # ax.set_title('Aquatic Sensor Network with Shaded Cube (Depth: 0 to -100m)')
        ax.set_box_aspect([1, 1, 0.8])  # 根据实际效果调整数值

        # 调整子图边距
        plt.tight_layout(pad=0)
        # 定义立方体的顶点（深度范围从 0 到 -100）
        cube_vertices = np.array([
            [0, 0, 0],  # 前左下
            [1000, 0, 0],  # 前右下
            [1000, 1000, 0],  # 前右上
            [0, 1000, 0],  # 前左上
            [0, 0, -1000],  # 后左下
            [1000, 0, -1000],  # 后右下
            [1000, 1000, -1000],  # 后右上
            [0, 1000, -100]  # 后左上
        ])

        # 定义立方体的面（每个面由4个顶点组成）
        cube_faces = [
            [cube_vertices[0], cube_vertices[1], cube_vertices[2], cube_vertices[3]],  # 前面
            [cube_vertices[4], cube_vertices[5], cube_vertices[6], cube_vertices[7]],  # 后面
            [cube_vertices[0], cube_vertices[1], cube_vertices[5], cube_vertices[4]],  # 左侧
            [cube_vertices[2], cube_vertices[3], cube_vertices[7], cube_vertices[6]],  # 右侧
            [cube_vertices[0], cube_vertices[3], cube_vertices[7], cube_vertices[4]],  # 底面
            [cube_vertices[1], cube_vertices[2], cube_vertices[6], cube_vertices[5]]  # 顶面
        ]
        # 绘制浅蓝色的立方体（半透明）
        light_blue = [0.5, 0.8, 0.9]  # 浅蓝色
        for face in cube_faces:
            poly = Poly3DCollection([face], color=light_blue, alpha=0.2, linewidths=0.1, edgecolors='black')
            ax.add_collection3d(poly)

        # 节点坐标
        x = np.zeros(self.n_node)
        y = np.zeros(self.n_node)
        z = np.zeros(self.n_node)
        for i in range(self.n_node):
            x[i] = self.nodes[i].position[0] * 3 + 200
            y[i] = self.nodes[i].position[1] * 3 + 200
            z[i] = self.nodes[i].position[2] * 3 - 900


        # 绘制水下传感器节点
        sensor_nodes = ax.scatter(x, y, z, c='blue', s=40, marker='o', label='Sensor nodes')

        # 绘制sink节点（水面）
        sink_x, sink_y = 500, 500  # 中心位置
        sink_z = 0
        ax.scatter(sink_x, sink_y, sink_z, c='red', s=100, marker='^', label='Cluster head')

        # 添加图例
        ax.legend()

        # 调整视角
        ax.view_init(elev=20, azim=-55)
        # plt.xticks(fontsize=18)
        # plt.yticks(fontsize=18)
        # plt.zticks(fontsize=18)

        # # 显示图形

        plt.show()
        # plt.savefig('Fig6.2.png', dpi=200)


