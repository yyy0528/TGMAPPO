import math
import random

import numpy as np
from .buoy import buoy_net, buoy_node, data_packet
import matplotlib.pyplot as plt


# random.seed(123)


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

    # 观测(智能体网络的输入)在强化学习当中好像只起到了加速收敛的作用
    # 这里需要考虑怎么记录数据包的路径
    def step(self, action):
        #
        # if self.step_number % (2*self.buoy_number) == 0 and self.packet_id < (5*self.buoy_number):
        #
        #     self.packet_generation()
        # if self.step_number % 15 == 0:
        #     print("move", self.step_number/15)
        #     self.move()
        self.step_number += 1

        done = False
        sender_id = action[0]
        receiver_id = action[1]
        sender = self.buoys[sender_id]

        receiver = self.buoys[receiver_id]
        idx = 0
        for neighbor_id in sender.neighbor:
            if neighbor_id == receiver.id:
                break
            else:
                idx += 1

        sender.send_number[idx] += 1
        rate = random.random()
        is_adj = True
        dist = math.sqrt(
            math.pow((sender.get_position()[0] - receiver.get_position()[0]), 2) + math.pow(
                (sender.get_position()[1] - receiver.get_position()[1]), 2))
        # 欧式距离
        # + math.pow(
        #                     (self.buoys[i].get_position()[2] - self.buoys[j].get_position()[2]), 2)
        if dist > 1.5:
            is_adj =False
            # print(f"-----------------------------------------is_adj:{is_adj}")


        if rate >= receiver.lost_rate and is_adj:
            receiver.packet += 1
            sender.packet -= 1
            sender.success_number[idx] += 1
            reach = True

            if sender.send_number[idx] >= 10:
                s_rate = sender.success_rate_neighbor[idx]

                sender.success_rate_neighbor[idx] = (s_rate + sender.success_number[idx]/10)/2
                sender.send_number[idx] = 0
                # sender.success_rate_neighbor[idx] = 0
                sender.success_number[idx] = 0
        else:
            reach = False
            if sender.send_number[idx] >= 10:
                s_rate = sender.success_rate_neighbor[idx]
                sender.success_rate_neighbor[idx] = (s_rate + sender.success_number[idx]/10)/2
                sender.send_number[idx] = 0
                # sender.success_rate_neighbor[idx] = 0
                sender.success_number[idx] = 0

        if reach:
            # 获取通信质量的奖励
            obs_receiver = self.get_obs(receiver_id)
            receiver_lost_rate = obs_receiver["lost_rate"]

            average = sum(receiver_lost_rate)/len(receiver_lost_rate)
            r_lost_rate = 1/average

            # 获取距离上的奖励
            pos_s = sender.position
            pos_r = receiver.position

            dis_s = math.sqrt(math.pow(pos_s[0] - self.sink[0], 2) + math.pow(pos_s[1] - self.sink[1], 2))
            dis_r = math.sqrt(math.pow(pos_r[0] - self.sink[0], 2) + math.pow(pos_r[1] - self.sink[1], 2))


            r_d = (dis_s - dis_r) / (2 * dis_s)


            reward = r_d + r_lost_rate




        else:

            reward = -0.5



        return reward, done, reach

    def render(self):
        """Renders the environment."""

        buoys = self.buoys
        x_coords = [buoy.position[0] for buoy in buoys if buoy.position is not None]
        y_coords = [buoy.position[1] for buoy in buoys if buoy.position is not None]
        # 画散点图
        plt.scatter(x_coords, y_coords)

        # 设置图表标题和坐标轴标签
        plt.title('buoy_position')
        plt.xlabel('X')
        plt.ylabel('Y')
        # 显示图表
        plt.show()

    # 每一个浮标对应一个obs
    def get_obs(self, buoy_id):
        buoy = self.buoys[buoy_id]
        neighbor_id = buoy.get_neighbor()
        neighbors = []
        # obs_energy = []

        obs_dis = []
        obs_lost_rate = buoy.success_rate_neighbor
        for agent_id in neighbor_id:
            neighbors.append(self.buoys[agent_id])
        for neighbor in neighbors:
            # obs_energy.append(neighbor.energy)

            pos = neighbor.position

            obs_dis.append(pos[0])
            obs_dis.append(pos[1])

        obs_dis.append(self.sink[0])
        obs_dis.append(self.sink[1])

        obs = {
            # "energy": obs_energy,
            "lost_rate": obs_lost_rate,
            "dis": obs_dis
        }
        return obs

    def get_obs_list(self, buoy_id):
        buoy = self.buoys[buoy_id]
        neighbor_id = buoy.get_neighbor()
        neighbors = []
        # obs_energy = []
        obs_dis = []
        obs_lost_rate = buoy.success_rate_neighbor
        for agent_id in neighbor_id:
            neighbors.append(self.buoys[agent_id])
        for neighbor in neighbors:
            # obs_energy.append(neighbor.energy)

            pos = neighbor.position

            obs_dis.append(pos[0])
            obs_dis.append(pos[1])

        obs_dis.append(self.sink[0])
        obs_dis.append(self.sink[1])

        obs = obs_lost_rate + obs_dis

        return obs

    def get_n_action(self, buoy_id):
        buoy = self.buoys[buoy_id]
        n = len(buoy.get_neighbor())
        return n

    def get_n_obs(self, buoy_id):
        obs = self.get_obs_list(buoy_id)

        n = len(obs)
        return n

    def get_remaining_energy(self):

        energy = np.zeros(len(self.buoys))
        i = 0
        for buoy in self.buoys:
            energy[i] = buoy.energy
            i += 1

        mean = np.mean(energy)
        std_dev = np.std(energy)
        return mean, std_dev


    def get_net_info(self):
        info = []
        for buoy in self.buoys:
            info.append(buoy.get_info())

        return info

    def get_net_info_id(self, net_id):
        net = self.buoy_net[net_id]
        info = net.get_info()

        return info

    def get_packet(self, buoy_id):

        return self.buoys[buoy_id].packet

    def get_avail_obs(self, buoy_id):

        buoy = self.buoys[buoy_id]
        neighbor_id = buoy.get_neighbor_two()
        neighbor = [0]*len(neighbor_id)
        i = 0
        for n_id in neighbor_id:
            n_buoy = self.buoys[n_id]
            dist = math.sqrt(
                math.pow((buoy.get_position()[0] - n_buoy.get_position()[0]), 2) + math.pow(
                    (buoy.get_position()[1] - n_buoy.get_position()[1]), 2))
            if dist <= 1.5:
                neighbor[i] = 1
            i += 1
        neighbor = neighbor + neighbor
        return neighbor

    def get_avail_action(self, buoy_id):

        buoy = self.buoys[buoy_id]
        neighbor_id = buoy.get_neighbor_two()
        neighbor = [0]*len(neighbor_id)
        i = 0
        for n_id in neighbor_id:
            n_buoy = self.buoys[n_id]
            dist = math.sqrt(
                math.pow((buoy.get_position()[0] - n_buoy.get_position()[0]), 2) + math.pow(
                    (buoy.get_position()[1] - n_buoy.get_position()[1]), 2))
            if dist <= 1.5:
                neighbor[i] = 1
            i += 1

        return neighbor

    def packet_generation(self):

        np.random.seed(2568)
        for buoy in self.buoys:
            if buoy.if_sink:
                continue

            random_number = np.random.poisson(lam=self.lambda_val)
            # if random_number == 0:
            #     random_number = 1
            buoy.packet += random_number
            for i in range(random_number):
                packet = data_packet(self.packet_id, buoy.id, self.sink_id)
                self.packets.append(packet)
                self.packet_id += 1


    def packet_generation_one(self, number):
        buoy = self.buoys[0]
        buoy.packet += number
        for i in range(number):
            packet = data_packet(self.packet_id, buoy.id, self.sink_id)
            self.packets.append(packet)
            self.packet_id += 1


    def move(self):
        self.buoy_net.move()
