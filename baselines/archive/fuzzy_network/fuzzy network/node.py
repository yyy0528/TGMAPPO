import copy
import math
import random
import numpy as np
from collections import deque

from ceng_current import current_move

NUM = 100



class DequeQueue:
    def __init__(self):
        self.items = deque()

    def enqueue(self, item):
        """添加一个元素到队列尾部"""
        self.items.append(item)

    def dequeue(self):
        """从队列头部移除一个元素并返回它"""
        if not self.is_empty():
            return self.items.popleft()
        else:
            raise IndexError("dequeue from an empty queue")

    def peek(self):
        """返回队列头部的元素但不移除它"""
        if not self.is_empty():
            return self.items[0]
        else:
            raise IndexError("peek from an empty queue")

    def is_empty(self):
        """检查队列是否为空"""
        return len(self.items) == 0

    def size(self):
        """返回队列中的元素数量"""
        return len(self.items)

    def show_elements(self):
        """打印队列中的所有元素"""
        print("队列中的元素:", list(self.items))

    def clear(self):
        """清除队列中的所有元素"""
        self.items.clear()


class Node:
    def __init__(self):
        self.node_id = None
        self.energy = None
        self.position = None
        self.neighbor = []
        self.n_packet = 0
        self.n_neighbor = 0
        self.packet = DequeQueue()
        self.if_sink = False
        self.trajectory = None


        self.original_position = []

    def initialize(self, node_id, energy, position):

        self.node_id = node_id
        self.energy = energy
        self.position = position
        self.original_position = position
        self.original_energy = energy



    def get_info(self):
        info = {'id': self.node_id, 'energy': self.energy, 'position': self.position}
        return info

    def send(self, packet_length, pos):
        self.n_packet -= 1
        r = math.sqrt(math.pow(pos[0] - self.position[0], 2) + math.pow(pos[1] - self.position[1], 2) + math.pow(pos[2] - self.position[2], 2))

        r = r/100

        mu_rel = 0.001  # Transmission power factor μre
        mu_tr = 0.001  # Transmission power factor μtr
        f = 10  # Carrier frequency f (kHz)
        l = packet_length
        term1 = 0.11 * (f ** 2) / (1 + f ** 2)
        term2 = 44 * (f ** 2) / (4100 + f ** 2)
        term3 = 2.75e-4 * (f ** 2)
        d_f = term1 + term2 + term3 + 0.003

        # 计算公式(5)：a
        a = 10 ** (d_f / 10)


        # 计算公式(4)：E_tr(r)
        E_tr_r = mu_tr * (r ** 1.5) * (a ** r) * l

        self.energy -= E_tr_r
        packet_id = self.packet.dequeue()
        return packet_id

    # 如果成功接收到数据，返回True
    def receive(self, packet_id, packet_length):
        self.packet.enqueue(packet_id)

        mu_rel = 0.001  # Transmission power factor μre
        l = packet_length

        # 计算公式(3)：E_re
        E_re = mu_rel * l



        self.energy -= E_re

    def get_position(self):
        return self.position

    def set_neighbor(self, nb_id):
        self.neighbor.append(nb_id)

    def delete_neighbor(self):
        self.neighbor = []

    def get_neighbor(self):
        return self.neighbor

    def node_refresh(self):
        # if self.if_sink:
            # print(self.neighbor)

        self.n_packet = 0
        self.packet.clear()
        self.energy = self.original_energy

        # print(self.original_energy)
        # if self.position==self.original_position:
        #     print("error")
        # if self.node_id == 0:
        #     print(self.position)
        self.position = self.original_position.copy()


        # if self.node_id == 0:
        #     print(self.move_x[0])
        # if self.node_id == 0:
        #     print(self.move_x)
        #     print(self.move_y)

    def update_neighbor(self):
        pass

class DataPacket:
    def __init__(self, packet_id, source_id, destination):
        self.packet_id = packet_id
        self.current_node = source_id  # 当前位置
        self.last_pos = source_id  # 判断是否被传输,上一个位置
        self.stay_time = 1  # 停留时间
        self.memories = []
        self.route = []
        self.first_come = 0
        self.life = 100
        self.route.append([0, source_id])
        # self.route.append(source_id)
        self.destination = destination
        self.if_sink = False
        self.q_time = []
        self.dy = 0

    def transmit(self, next_hop):
        self.route.append([self.dy + 1, next_hop])
        # self.route.append(next_hop)
        self.q_time.append(self.stay_time)
        self.current_node = next_hop
        if next_hop == self.destination:
            self.if_sink = True
            self.q_time.append(0)
        else:
            self.if_sink = False

    def add_memory(self, memory_id):
        self.memories.append(memory_id)


class Network:
    def __init__(self, nodes):
        self.nodes = []
        self.nodes_pos = []
        self.n_node = len(nodes)

        self.num_steps = 160
        self.trajectorys = None
        self.initialize_node(nodes)
        self.adj_matrix = self.build_adjacency_matrix()
        self.initialize_neighbor()
        self.move_time = 0

        # 注意！！！！


    def initialize_node(self, nodes):
        for i in range(self.n_node):
            node = Node()
            node.initialize(i, nodes[i][0], nodes[i][1])
            self.nodes.append(node)
            self.nodes_pos.append([node.position[0]/100, node.position[1]/100, node.position[2]/100])
            # self.nodes_pos.append(node.position[0])
            # self.nodes_pos.append([node.position[0], node.position[1], node.position[2]])
        self.trajectorys = self.get_trajectory()
    def reset(self):
        self.move_time = 0
        for node in self.nodes:
            node.node_refresh()
            node.delete_neighbor()
        self.adj_matrix = self.build_adjacency_matrix()

        self.initialize_neighbor()

        # 这个是挺关键的地方
        self.move_time = 0

    # 一跳邻居
    def build_adjacency_matrix(self):

        adj_matrix = np.zeros((self.n_node, self.n_node))
        # 只要通信范围内的节点的距离，这样通过列的序号就得到这个邻居的编号。
        for i in range(self.n_node):
            cc = 0  # 计数
            for j in range(self.n_node):
                dist = math.sqrt(
                    math.pow((self.nodes[i].get_position()[0] - self.nodes[j].get_position()[0]), 2) + math.pow(
                        (self.nodes[i].get_position()[1] - self.nodes[j].get_position()[1]), 2) + math.pow(self.nodes[i].get_position()[2] - self.nodes[j].get_position()[2], 2))
                # 欧式距离
                # + math.pow(
                #                     (self.nodes[i].get_position()[2] - self.nodes[j].get_position()[2]), 2)
                if dist <= 160:  # 150m
                    adj_matrix[i][j] = dist  # 只要通信范围内的节点的距离，这样通过列的序号就得到这个邻居的编号。
                    cc += 1
                else:
                    adj_matrix[i][j] = 0
            # self.count_NB.append(cc - 1)  # 统计邻居数量，减去把自己当邻居的情况
        '''if(math.sqrt(math.pow((x[i]-250),2)+math.pow((y[i]-250),2)+math.pow(z[i]-250,2)))<=self.r:
                candidate_of_cluster_heads.append(i) '''
        # if np.array(self.count_NB).all() >= 1:  # 每个节点的邻居数至少为2时，则终止循环，为了避免后续因邻居过少而报错
        #     self.do = False
        return adj_matrix




    def initialize_neighbor(self):
        for node in self.nodes:
            for i in range(self.n_node):
                if self.adj_matrix[node.node_id][i] > 0:
                    node.set_neighbor(i)


    def get_nodes(self):
        return self.nodes


    def transmit(self, sender, receiver):
        pass

    # def move(self):
    #     for node in self.nodes:
    #         if node.if_sink:
    #             continue
    #         # move_step_x = random.normalvariate(0, 0.01)
    #         # move_step_y = random.normalvariate(0, 0.01)
    #         # move_step_z = random.normalvariate(0, 0.01)
    #         # if node.node_id == 0:
    #         #     print(self.move_time, move_step_x)
    #         node.move(self.move_time)
    #         # node.move(move_step_x, move_step_y, move_step_z)
    #
    #         node.delete_neighbor()
    #     self.move_time = self.move_time + 1
    #     if self.move_time == 10:
    #         self.move_time = 0
    #     # print(self.move_time)
    #     self.adj_matrix = self.build_adjacency_matrix()
    #     self.initialize_neighbor()


    def move(self):
        for node in self.nodes:

            # 以正太分布为基础的那个洋流模型
            # node.position[0] = self.trajectorys[0][node.node_id][self.move_time+1]
            # node.position[1] = self.trajectorys[1][node.node_id][self.move_time+1]
            # node.position[2] = self.trajectorys[2][node.node_id][self.move_time+1]
            # node.move(move_step_x, move_step_y, move_step_z)

            # 不带随机数的层流模型
            node.position[0] = self.trajectorys[node.node_id][self.move_time+1][0] * 100
            node.position[1] = self.trajectorys[node.node_id][self.move_time+1][1] * 100
            node.position[2] = self.trajectorys[node.node_id][self.move_time+1][2] * 100


            node.delete_neighbor()
        self.move_time = self.move_time + 1

        # 这个东西一定要跟 episode的长度相对应才行
        if self.move_time == self.num_steps -1:
            self.move_time = 0
        # print(self.move_time)
        self.adj_matrix = self.build_adjacency_matrix()
        self.initialize_neighbor()

    def get_trajectory(self):

        self.trajectorys = current_move(self.nodes_pos, 0.1, self.num_steps)

        xxx = self.trajectorys[0]

        return self.trajectorys


    def get_edge_index(self):

        edge_index = []
        for i in range(self.n_node):
            for j in range(self.n_node):
                dist = math.sqrt(
                    math.pow((self.nodes[i].get_position()[0] - self.nodes[j].get_position()[0]), 2) + math.pow(
                        (self.nodes[i].get_position()[1] - self.nodes[j].get_position()[1]), 2) + math.pow(self.nodes[i].get_position()[2] - self.nodes[j].get_position()[2], 2))
                # 欧式距离
                # + math.pow(
                #                     (self.nodes[i].get_position()[2] - self.nodes[j].get_position()[2]), 2)
                if dist <= 160:  # 150m
                    edge_index.append([i,j])

        return edge_index

    def get_node_feat(self):
        node_feat = []
        for node in self.nodes:
            # 1. 提取原始数据
            node_id = node['id']  # 节点ID（可选，用于调试）
            x = node.position[0]  # x坐标
            y = node.position[1]   # y坐标
            z = node.position[2]   # z坐标
            energy = node.energy # 剩余能量

            # 2. 归一化三维坐标到[0,1]
            # x坐标归一化

            norm_x = x/100
            # y坐标归一化

            norm_y = y/100
            # z坐标归一化（若为2D网络，z可固定为0或忽略）

            norm_z = z/100

            # 3. 归一化剩余能量到[0,1]
            norm_energy = energy / 100  # 避免能量超过最大值（如充电场景）

            # 4. 组合特征（顺序：x, y, z, 能量）
            feat = [norm_x, norm_y, norm_z, norm_energy]
            node_feat.append(feat)

        # 转换为PyTorch张量（float32类型，GNN常用格式）
        # return torch.tensor(node_feat, dtype=torch.float32)

        return node_feat