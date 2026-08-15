import logging
import matplotlib.pyplot as plt
import numpy as np
# 设置日志记录
# policy_name= ""
# logging.basicConfig(filename=f'{policy_name}_training_log.txt', level=logging.INFO, format='%(message)s')
import logging
import os
from .experiment_io import MetricCsvWriter, moving_buoy_ids

policy_name = "VDN"
class Runner:
    def __init__(self, agents, env, algorithm_name="VDN", episodes=1000,
                 episode_length=80, num_moving_buoys=25,
                 output_root="results", grid_size=5):
        global policy_name
        policy_name = algorithm_name.replace("-", "_")
        self.run_dir = os.path.join(
            output_root, "moves_{}".format(num_moving_buoys)
        )
        log_dir = os.path.join(self.run_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.join(log_dir, f"{policy_name}_training_log.txt")
        logging.getLogger().handlers.clear()
        logging.basicConfig(
            filename=log_file_path,
            level=logging.INFO,
            format="%(message)s",
        )
        self.reward = None
        self.time = None
        self.agents = agents
        self.env = env
        self.step = 0
        self.done = False
        self.len = []
        self.q_time = []
        self.route = []
        self.episodes = episodes
        self.episode_length = episode_length
        self.moving_buoy_ids = moving_buoy_ids(
            len(self.agents), num_moving_buoys
        )
        self.metric_writer = MetricCsvWriter(
            output_root, num_moving_buoys, algorithm_name
        )



    def run(self):
        rate_list = []
        average_hops_list = []
        episode_list = []
        average_q_time_list = []
        # step_list = []

        for episode in range(self.episodes):
            # initial
            self.len = []
            self.env.reset()
            self.done = False
            self.step = 0
            self.route = []
            self.reward = 0
            self.env.packet_generation()
            # print("done")
            while True:
                for agent in self.agents:
                    buoy = agent.buoy
                    packets = self.env.packets
                    # 判断是否是空洞节点
                    if buoy.neighbor == []:
                        continue
                    # 判断是否有待转发的数据包
                    if buoy.packet == 0:
                        packets = self.env.packets

                        # 奖励值回溯
                        to_del = []
                        k = 0
                        index = -1
                        for packet in packets:
                            index += 1

                            if_sink = packet.if_sink

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
                                                r = r + (r_q_time + (1 / len(route)) * 10)*10
                                                s_ = trans[3]
                                                new_done = trans[4]
                                                agent.RL.store_transition(s, a, r, s_, new_done, True)
                                                self.reward += (r_q_time + (1 / len(route)) * 10)*10

                                                break
                                        packet.memories[i] = -1
                                        break

                                if packet.change_n == len(route) - 1:
                                    self.len.append(len(route))
                                    packet.change_n += 1

                                    to_del.append(index)

                        for i in sorted(to_del, reverse=True):
                            del packets[i]
                        continue

                    if buoy.if_sink:

                        continue

                    observation = self.env.get_obs_list(buoy.id)

                    neighbor = buoy.neighbor
                    if self.env.sink_id in neighbor:
                        action = [buoy.id, self.env.sink_id]
                        action_origin = self.env.sink_id
                    else:

                        action_origin = agent.RL.choose_action(observation)

                        # print(f"action:{action_origin}")
                        action_origin = action_origin % len(buoy.neighbor)
                        # print(f"after::::action:{action_origin}")
                        action = [buoy.id, buoy.neighbor[action_origin]]

                        agent.step += 1

                    reward, done, reach = self.env.step(action)

                    observation_ = self.env.get_obs_list(buoy.id)

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
                                    packet.stay_time = 1
                                else:
                                    packet.stay_time += 1
                        if_sink = packet.if_sink
                        if if_sink:

                            buffer = agent.RL.replay_buffer.get_buffer()
                            route = packet.route

                            for i in range(len(route)):
                                if route[i] == agent.buoy.id:
                                    memory = packet.memories[i]
                                    packet.change_n += 1
                                    r_q_time = -(packet.q_time[i+1] / sum(packet.q_time)) * 5

                                    for trans in buffer:
                                        if memory == trans[5]:
                                            s = trans[0]
                                            a = trans[1]
                                            r = trans[2]
                                            # 计算补充奖励
                                            r = r + (r_q_time + (1 / len(route)) * 10)*10
                                            s_ = trans[3]
                                            new_done = trans[4]
                                            # buffer.remove(trans)
                                            agent.RL.store_transition(s, a, r, s_, new_done, True)
                                            self.reward += (r_q_time + (1 / len(route)) * 10)*10

                                            break
                                    packet.memories[i] = -1
                                    break
                            if packet.change_n == len(route)-1:
                                packet.change_n += 1
                                self.len.append(len(route))
                                self.route.append(route)
                                self.q_time.append(sum(packet.q_time))
                                to_del.append(index)

                    # 删除已经到达汇聚节点数据包，减少循环次数
                    for i in sorted(to_del, reverse=True):
                        del packets[i]

                    # 超过200条transition之后每隔5步学习一次
                    if (agent.step > 200) and (agent.step % 5 == 0):

                        agent.RL.learn()

                self.step += 1

                # 定期改变节点间的丢包率
                if (self.step > 3) and (self.step % 20 == 0):
                    self.env.buoy_net.reset_lost_rate()


                if self.step > self.episode_length:

                    if (episode+1)%1 ==0:

                        # 新增：所有浮标完成动作后，统一调用move()
                        for agent in self.agents:
                            if agent.buoy.id in self.moving_buoy_ids:
                                agent.buoy.move()  # 每个浮标执行移动
                                # print(1)
                        # 移动后可能需要重新计算邻居关系（如果位置变化影响邻居判断）
                        # 例如：self.env.update_neighbors()  # 假设环境有更新邻居的方法

                        # 移动完v成后，显式调用update_topology()更新距离和邻居
                        # self.env.buoy_net.update_topology((np.zeros((25, 25), dtype=np.float32)))

                    break

            # 计算交付率、平均跳数、平均时延
            rate = len(self.len) / self.env.packet_id if self.env.packet_id else 0
            loss_num = self.env.packet_id - len(self.len)
            average_hops = ((sum(self.len) + self.episode_length * loss_num)
                            / self.env.packet_id if self.env.packet_id else 0)
            # average_hops = sum(self.len) / len(self.len)
            average_q_time = sum(self.q_time) / len(self.q_time) if self.q_time else 0
            print(
                f"episode: {episode:4d} | "
                f"rate: {rate:7.4f} | "
                f"average_hops: {average_hops:8.3f} | "
                f"average_q_time: {average_q_time:8.3f}"
            )
            # 记录日志
            logging.info(f'{episode},{rate},{average_hops},{average_q_time}')
            self.metric_writer.write(
                episode, rate, average_hops, average_q_time
            )

            # ===== 保存用于绘图 =====
            episode_list.append(episode)
            rate_list.append(rate)
            average_hops_list.append(average_hops)
            average_q_time_list.append(average_q_time)
            # step_list.append(self.step)

            # 绘制图表
        # ================= 绘图 =================
        plt.figure(figsize=(10, 8))

        # 上图：episode vs rate
        plt.subplot(3, 1, 1)
        plt.plot(episode_list, rate_list)
        plt.xlabel("Episode")
        plt.ylabel("Rate")
        plt.title("Delivery Rate vs Episode")

        # 下图：episode vs average_hops
        plt.subplot(3, 1, 2)
        plt.plot(episode_list, average_hops_list)
        plt.xlabel("Episode")
        plt.ylabel("Average Hops")
        plt.title("Average Hops vs Episode")

        # 下图：episode vs average_hops
        plt.subplot(3, 1, 3)
        plt.plot(episode_list, average_q_time_list )
        plt.xlabel("Episode")
        plt.ylabel("Average Times")
        plt.title("Average Times vs Episode")

        plt.tight_layout()
        # ✅ 保存图片（一定要在 show 之前）
        png_dir = os.path.join(self.run_dir, "plots")
        os.makedirs(png_dir, exist_ok=True)
        save_path =os.path.join(png_dir,f"{policy_name}_rate_and_hops_vs_episode.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")


        plt.show()
        self.metric_writer.close()






