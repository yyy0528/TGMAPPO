import math
import random
import numpy as np
from buoy import buoy_net, buoy_node, data_packet
import matplotlib.pyplot as plt

class buoy_environment:
    def __init__(self, buoys):
        self.buoy_number = len(buoys)
        self.sink = None  # sink节点坐标
        self.sink_id = None
        self.buoys_info = buoys  # 创建对象的信息
        self.buoys = None  # 浮标对象
        self.buoy_net = None
        self.lambda_val = 2
        self.packets = []
        self.packet_id = 0
        self.route_kong = True
        self.step_number = 0

    def build_env(self):
        self.step_number = 0
        self.buoy_net = buoy_net(self.buoys_info)

        self.buoys = self.buoy_net.get_buoys()
        self.sink_id = len(self.buoys) - 1
        buoy_sink = self.buoys[self.sink_id]
        self.sink = buoy_sink.position

        buoy_sink.if_sink = True
        buoy_sink.energy = 9999
        self.buoys.append(buoy_sink)
        observations = []

        info = self.get_net_info()

        return observations, info

    def reset(self):
        for buoy in self.buoys:
            buoy.node_refresh()
        self.packets = []
        self.packet_id = 0
        self.step_number = 0

    def step(self, action):
        self.step_number += 1
        done = False
        sender_id = action[0]
        receiver_id = action[1]
        sender = self.buoys[sender_id]
        receiver = self.buoys[receiver_id]

        sender.send_number[sender_id] += 1  # 增加发送次数

        # 根据距离计算信号强度和成功率
        distance = self.calculate_distance(sender.position, receiver.position)

        signal_strength = self.calculate_signal_strength(distance)  # 信号强度
        success_rate = signal_strength * (1 - receiver.lost_rate)  # 调整的成功概率

        # 模拟包的传递
        rate = random.random()
        reach = rate < success_rate  # 根据成功率判断是否成功

        if reach:
            sender.packet -= 1
            sender.success_number[sender_id] += 1  # 记录成功
            # 确认机制
            confirmation_rate = random.random()
            if confirmation_rate < success_rate:  # 确认接收
                receiver.packet += 1
                sender.confirmations[sender_id] += 1  # 记录确认
            else:  # 确认失败，重发
                self.try_resend(sender, receiver, sender_id)

        # 计算奖励
        reward, selfReward = self.calculate_rewards(sender, receiver, reach)
        dis_reward = self.calculate_distance_reward(sender, receiver)
        reward += dis_reward

        return reward, selfReward ,done, reach

    def try_resend(self, sender, receiver, sender_id):
        # 尝试重发包
        resending_rate = random.random()
        receiver.packet += 1
        sender.confirmations[sender_id] += 1  # 记录成功重发

    def calculate_rewards(self, sender, receiver, reach):
        self_reward = 1 if reach else -1
        # 计算负载均衡奖励
        total_packets = sum(buoy.packet for buoy in self.buoys)
        mean_load = total_packets / self.buoy_number if self.buoy_number > 0 else 0
        sender_load = sender.packet
        receiver_load = receiver.packet
        load_diff_sender = mean_load - sender_load
        load_diff_receiver = mean_load - receiver_load

        # 计算负载均衡奖励，并添加惩罚
        load_balancing_reward = (load_diff_sender + load_diff_receiver) / 2  # 平均负载差
        if abs(load_diff_sender) > 3 and abs(load_diff_receiver) > 3:  # 如果负载均衡过度
            load_balancing_reward -= 1  # 扣除奖励
        if(load_balancing_reward != 0):
            load_balancing_reward = 1/load_balancing_reward

        # 通信损失的惩罚
        communication_loss_penalty = (sender.success_number[sender.id] - sender.packet) * 1  # 每次失败扣1分
        if sender.packet > 0:
            communication_loss_penalty = min(communication_loss_penalty, sender.packet)  # 确保惩罚不会超过发送的包数量

        if reach:
            r_distance = self.calculate_distance_reward(sender, receiver)
            self_reward = self_reward + r_distance - communication_loss_penalty
        else:
            self_reward = self_reward - 0.5 - communication_loss_penalty  # 失败的惩罚加上通信损失惩罚

        return load_balancing_reward, self_reward

    def calculate_distance(self, pos1, pos2):
        """计算两点之间的距离"""
        return math.sqrt((pos1[0][0] - pos2[0][0]) ** 2 + (pos1[0][1] - pos2[0][1]) ** 2 + (pos1[0][2] - pos2[0][2]) ** 2)

    def calculate_signal_strength(self, distance):
        """计算信号强度"""
        if distance == 0:
            return 1.0  # 相同位置，信号强度最大
        return max(0, 1 - (distance / 1000))  # 信号强度

    def calculate_distance_reward(self, sender, receiver):
        # 计算距离奖励
        dis_s = self.calculate_distance(sender.position, self.sink)
        dis_r = self.calculate_distance(receiver.position, self.sink)
        return (dis_s - dis_r) / (2 * dis_s)

    def get_obs(self, buoy_id):
        buoy = self.buoys[buoy_id]
        neighbor_id = buoy.get_neighbor()
        neighbors = []
        obs_dis = []
        obs_lost_rate = buoy.success_rate_neighbor
        for agent_id in neighbor_id:
            neighbors.append(self.buoys[agent_id])
        for neighbor in neighbors:
            pos = neighbor.position
            obs_dis.append(pos[0][0])
            obs_dis.append(pos[0][1])
            obs_dis.append(pos[0][2])
        obs_dis.append(self.sink[0][0])
        obs_dis.append(self.sink[0][1])
        obs_dis.append(self.sink[0][2])
        obs = {
            "lost_rate": obs_lost_rate,
            "dis": obs_dis
        }
        return obs

    def get_obs_list(self, buoy_id):
        buoy = self.buoys[buoy_id]
        neighbor_id = buoy.get_neighbor()
        neighbors = []
        obs_dis = []
        obs_lost_rate = buoy.success_rate_neighbor
        for agent_id in neighbor_id:
            neighbors.append(self.buoys[agent_id])
        for neighbor in neighbors:
            pos = neighbor.position
            obs_dis.append(pos[0][0])
            obs_dis.append(pos[0][1])
            obs_dis.append(pos[0][2])
        obs_dis.append(self.sink[0][0])
        obs_dis.append(self.sink[0][1])
        obs_dis.append(self.sink[0][2])

        obs = obs_lost_rate + obs_dis

        return obs

    def get_n_action(self, buoy_id):
        buoy = self.buoys[buoy_id]
        return len(buoy.get_neighbor())

    def get_n_obs(self, buoy_id):
        obs = self.get_obs_list(buoy_id)
        return len(obs)

    def get_remaining_energy(self):
        energy = np.zeros(len(self.buoys))
        for i, buoy in enumerate(self.buoys):
            energy[i] = buoy.energy
        mean = np.mean(energy)
        std_dev = np.std(energy)
        return mean, std_dev

    def get_net_info(self):
        return [buoy.get_info() for buoy in self.buoys]

    def get_packet(self, buoy_id):
        return self.buoys[buoy_id].packet

    def move(self):
        self.buoy_net.move()

    def packet_generation(self):
        np.random.seed(2568)
        for buoy in self.buoys:
            if buoy.if_sink:
                continue
            random_number = np.random.poisson(lam=self.lambda_val)
            buoy.packet += random_number
            for _ in range(random_number):
                packet = data_packet(self.packet_id, buoy.id, self.sink_id)
                self.packets.append(packet)
                self.packet_id += 1

    def packet_generation_one(self, number):
        buoy = self.buoys[0]
        buoy.packet += number
        for _ in range(number):
            packet = data_packet(self.packet_id, buoy.id, self.sink_id)
            self.packets.append(packet)
            self.packet_id += 1
