import math


class Runner:
    def __init__(self, agents, env):
        self.reward = None
        self.time = None
        self.agents = agents
        self.env = env
        self.step = 0
        self.done = False
        self.len = []
        self.q_time = []
        self.route = []

    def run(self):
        for episode in range(10):
            # 初始化
            self.len = []
            self.env.reset()
            self.done = False
            self.step = 0
            self.route = []
            self.reward = 0
            self.reach = None
            self.env.packet_generation()
            print("done")
            while True:
                for agent in self.agents:
                    buoy = agent.buoy
                    packets = self.env.packets

                    # 检查是否是空洞节点
                    if buoy.neighbor == []:
                        continue

                    # 检查是否有待转发的数据包
                    if buoy.packet == 0:
                        packets = self.env.packets
                        to_del = []
                        k = 0
                        index = -1
                        for packet in packets:
                            index += 1
                            if_sink = packet.if_sink

                            # 处理到达汇聚节点的数据包
                            if if_sink:
                                buffer = agent.RL.replay_buffer.get_buffer()
                                route = packet.route
                                for i in range(len(route)):
                                    if route[i] == agent.buoy.id:
                                        r_q_time = -(packet.q_time[i+1] / sum(packet.q_time)) * 5
                                        memory = packet.memories[i]
                                        packet.change_n += 1
                                        for trans in buffer:
                                            if memory == trans[5]:
                                                s = trans[0]
                                                a = trans[1]
                                                r = trans[2]
                                                # 增加补充奖励
                                                r += (r_q_time + (1 / len(route)) * 10) * 10
                                                s_ = trans[3]
                                                new_done = trans[4]
                                                agent.RL.store_transition(s, a, r, s_, new_done, True)
                                                self.reward += (r_q_time + (1 / len(route)) * 10) * 10
                                                break
                                        packet.memories[i] = -1
                                        break

                                if packet.change_n == (len(route) - 1):
                                    #processes of ack and sending, adding the first node and multiply the average turns
                                    self.len.append(1+len(route)*0.2)
                                    packet.change_n += 1
                                    to_del.append(index)

                        for i in sorted(to_del, reverse=True):
                            del packets[i]
                        continue

                    if buoy.if_sink:
                        continue

                    # 获取当前观察状态
                    observation = self.env.get_obs_list(buoy.id)
                    neighbor = buoy.neighbor

                    # 选择动作
                    if self.env.sink_id in neighbor:
                        action = [buoy.id, self.env.sink_id]
                        action_origin = self.env.sink_id
                    else:
                        action_list = agent.RL.choose_action(observation)
                        action_origin = int(action_list[0]) if isinstance(action_list, list) else int(action_list)
                        action = [buoy.id, buoy.neighbor[action_origin]]
                        agent.step += 1

                    # 执行动作
                    reward, selfreward, done, reach = self.env.step(action)
                    observation_ = self.env.get_obs_list(buoy.id)

                    # 存储当前转移
                    memory_id = agent.RL.store_transition(observation, action_origin, reward, observation_, done, False)
                    self.reward += reward

                    # 奖励值回溯
                    to_del = []
                    k = 0
                    index = -1
                    for packet in packets:
                        index += 1
                        if_sink = packet.if_sink

                        if not if_sink:
                            if packet.agent_id == action[0] and reach:
                                if k == 0:
                                    k = 1
                                    packet.transmit(action[1])
                                    packet.add_memory(memory_id)
                                    packet.stay_time = 0.4
                                else:
                                    packet.stay_time += 0.4
                        if_sink = packet.if_sink
                        if if_sink:
                            buffer = agent.RL.replay_buffer.get_buffer()
                            route = packet.route

                            for i in range(len(route)):
                                if route[i] == agent.buoy.id:
                                    memory = packet.memories[i]
                                    packet.change_n += 1
                                    r_q_time = -(packet.q_time[i+1] / sum(packet.q_time))

                                    for trans in buffer:
                                        if memory == trans[5]:
                                            s = trans[0]
                                            a = trans[1]
                                            r = trans[2]
                                            # 计算补充奖励
                                            r += (r_q_time + (1 / len(route)) * 10) * 10
                                            s_ = trans[3]
                                            new_done = trans[4]
                                            agent.RL.store_transition(s, a, r, s_, new_done, True)
                                            self.reward += (r_q_time + (1 / len(route)) * 10) * 10
                                            break
                                    packet.memories[i] = -1
                                    break

                            if packet.change_n == (len(route) - 1):
                                packet.change_n += 1
                                #processes of ack and sending, adding the first node and multiply the average turns
                                self.len.append(1+len(route)*0.2)
                                self.route.append(route)
                                self.q_time.append(sum(packet.q_time))
                                to_del.append(index)

                    # 删除已经到达汇聚节点的数据包，减少循环次数
                    for i in sorted(to_del, reverse=True):
                        del packets[i]

                    # 超过200条transition之后每隔5步学习一次
                    if (agent.step > 200) and (agent.step % 5 == 0):
                        agent.RL.learn()

                self.step += 1

                # 定期改变节点间的丢包率
                if (self.step > 3) and (self.step % 20 == 0):
                    self.env.buoy_net.reset_lost_rate()

                # 设置最大步数
                if self.step > 120:
                    break

            # 计算交付率、平均跳数、平均时延
            rate = len(self.len) / self.env.packet_id
            average_hops = sum(self.len) / len(self.len)
            average_q_time = sum(self.q_time) / len(self.q_time)
            print('step', self.step)
            print('交付率 ', rate)
            print('均跳数 ', average_hops)
            print('均时延 ', average_q_time)
