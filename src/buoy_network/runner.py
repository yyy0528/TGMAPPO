import numpy as np
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
from datetime import datetime
import os
import time
import json
import logging
from datetime import datetime
import os
from .utils import setup_logger
from src.experiment_io import MetricCsvWriter, moving_buoy_ids
from .transition_data import TransitionRecorder, routing_actions
import random
import csv

logger = setup_logger(log_dir="outputs/TG-MAPPO/diagnostic_logs", log_name="buoy_2018")
class tmp_buffer:
    def __init__(self,memory_id:int):
        self.reward = 0
        self.memory_id = memory_id



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
        # 新增：存储所有episode的指标
        self.episode_rates = []  # 存储每个episode的交付率
        self.episode_hops = []  # 存储每个episode的平均跳数
        self.episode_indices = []  # 存储episode编号（横坐标）
        self.metric_writer = MetricCsvWriter(
            self.env.all_args.output_root,
            self.env.all_args.num_moving_buoys,
            "TG-MAPPO",
        )
        self.moving_buoy_ids = moving_buoy_ids(
            len(self.agents), self.env.all_args.num_moving_buoys
        )

        # 新增：日志配置
        self.run_dir = os.path.join(
            self.env.all_args.output_root,
            "moves_{}".format(self.env.all_args.num_moving_buoys),
        )
        self.log_dir = os.path.join(self.run_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)  # 创建日志目录
        self.pm_runtime = getattr(self.agents[0].RL, "pm_runtime", None)
        self.pm_stream = None
        if self.pm_runtime is not None:
            self.pm_stream = open(os.path.join(self.log_dir, "TG_PM_rewards.csv"), "x", newline="", encoding="utf-8")
            self.pm_writer = csv.DictWriter(self.pm_stream, fieldnames=("episode", "step", "reward_env", "phi_t",
                "phi_tp1", "reward_progress", "reward_train", "terminated", "rollout_truncated"))
            self.pm_writer.writeheader()
        self.data_exporter = None
        self.data_behavior = getattr(self.env.all_args, "data_behavior", "tg-mappo")
        self.data_rng = random.Random(getattr(self.env.all_args, "seed", 0))
        if getattr(self.env.all_args, "export_wm_data", False):
            self.data_exporter = TransitionRecorder(
                os.path.join(self.run_dir, "wm_data"), self.env.all_args,
                self.env.all_args.wm_export_max_steps,
                self.env.all_args.wm_export_max_bytes,
                self.env.all_args.wm_export_max_packets)
        log_file = os.path.join(self.log_dir, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

        # 配置日志：同时输出到文件和控制台
        self.logger = logging.getLogger(__name__ + "." + str(id(self)))
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self._log_handler = logging.FileHandler(log_file)
        self._log_handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(self._log_handler)
        with open(os.path.join(self.log_dir, "TG_MAPPO_environment.json"), "w", encoding="utf-8") as stream:
            json.dump({"implementation": "legacy_fixed_v1", "node_features": [
                "x", "y", "queue_fraction", "entity_type"],
                "action_schema": "global_node_wait_v1", "config": vars(self.env.all_args)},
                stream, indent=2)

    def run(self):
        try:
            return self._run()
        finally:
            if self.pm_stream is not None:
                self.pm_stream.close()
            if self.data_exporter is not None:
                self.data_exporter.close()
            self.metric_writer.close()
            self.logger.removeHandler(self._log_handler)
            self._log_handler.close()

    def _run(self):
        episodes = self.env.all_args.num_episodes
        added_value = np.zeros((len(self.agents),len(self.agents)),dtype= np.float32)
        penalty = -35
        for episode in range(episodes):
            # initial
            self.len = []
            self.q_time = []
            self.env.reset()
            self.done = False
            self.step = 0
            self.route = []
            self.reward = 0
            self.env.packet_generation()
            next_queue_order = self.env.packet_id
            self.agents[0].RL.warmup()
            if self.pm_runtime is not None:
                self.pm_runtime.start_episode(self.env, episode)
            if self.data_exporter is not None:
                self.data_exporter.start_episode(self.env, episode, added_value)
            self.memory_id =0
            self.agents[0].RL.policy.lr_decay(episode, episodes)
            temp_reward = 0
            memory_id_poll=[]

            #print("done")
            while self.step < self.env.all_args.episode_length:

                available_t = self.env.available_actions()
                service_packet_ids = {}
                for packet in sorted(self.env.packets, key=lambda queued: queued.queue_order):
                    if not packet.if_sink and packet.agent_id not in service_packet_ids:
                        service_packet_ids[packet.agent_id] = packet.id
                forwarded_packet_ids = set()

                # Sample actions
                #print("self_step",self.step)
                (
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                    actions_env,
                ) = self.agents[0].RL.collect(self.step)
                if self.data_behavior != "tg-mappo":
                    rule_actions, rule_logp = routing_actions(self.env, self.data_behavior, self.data_rng)
                    actions = np.asarray(rule_actions, dtype=np.int64)
                    action_log_probs = np.asarray(rule_logp, dtype=np.float32)
                    actions_env = np.squeeze(np.eye(self.env.buoy_number + 1)[actions], 2)
                for node in range(self.env.buoy_number):
                    action_index = int(actions[0][node][0])
                    if not 0 <= action_index <= self.env.buoy_number or not available_t[0, node, action_index]:
                        raise ValueError("Sampled action violates the current physical action mask")
                self.memory_id += 1
                if self.data_exporter is not None:
                    self.data_exporter.start_step(self.env, actions, action_log_probs,
                                                  values, service_packet_ids)
                actions_mask_list,  obs_list, agent_id_list, node_obs_list, adj_list, reward_list, done_list=[],[],[],[],[],[],[]
                #print(obs_list)
                tme_b = tmp_buffer(self.memory_id)
                #obs, agent_id, node_obs, adj, rewards, dones, infos = self.env.step(
                #    actions_env
                #)


                #print("values: ",values)
                #print("actions: ", actions)
                #print("action_log_probs: ", action_log_probs)
                #print("rnn_states: ", rnn_states)
                #print("rnn_states_critic: ", rnn_states_critic)
                #print("actions_env: ", actions_env)

                #print("obj: ", obs)
                #print("agent_id: ", agent_id)
                #print("nodes_obs: ", adj)
                #print("reward: ", rewards)
                #print("dones: ", dones)
                #print("infos: ", infos)
                packet_num_list  = []
                for i, agent_i in  enumerate(self.agents):
                    packet_num_list.append(agent_i.buoy.packet)


                added_value_1_flag = np.zeros((len(self.agents),len(self.agents)),dtype= np.float32)

                #print("packet_num_list", packet_num_list)
                #print("buffer reward", self.agents[0].RL.buffer.rewards[0] [0][0][0])
                for agent in self.agents:
                    buoy = agent.buoy
                    packets = self.env.packets
                    #print("packets_num:" , len(packets))
                    # 判断是否是空洞节点
                    if buoy.neighbor == []:
                        obs_n, agent_id_n, node_obs_n, adj_n, reward_n, done_n = self.env.step_one_agent(buoy.id, 0, False)
                        obs_list.append(obs_n[0])
                        agent_id_list.append(agent_id_n[0])
                        node_obs_list.append(node_obs_n[0])
                        adj_list.append(adj_n[0])
                        reward_list.append(reward_n[0])
                        done_list.append(done_n[0])
                        actions_mask_list.append([0])
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
                            #print("index",index)

                            if_sink = packet.if_sink

                            if if_sink:
                                buffer = self.agents[0].RL.buffer
                                route = packet.route
                                # print(f"route_141: {route}")
                                route = packet.route
                                # print(f"route长度: {len(route)}, route内容: {route}")
                                # print(f"q_time长度: {len(packet.q_time)}, q_time内容: {packet.q_time}")
                                for i in range(len(route)):
                                    if route[i] == agent.buoy.id:
                                        r_q_time = -(packet.q_time[i+1] / sum(packet.q_time)) * 5
                                        memory = packet.memories[i]
                                        memory_id_poll.append(memory)
                                        packet.change_n += 1

                                        if memory == tme_b.memory_id:
                                            reward_list.append(0)



                                        #print("buffer reward",buffer.rewards[0])
                                        # print("buffer reward", buffer.rewards[0])
                                        for index_tmp, buffer_memory_id in enumerate(buffer.memory_list):
                                            if memory == buffer_memory_id:
                                                # 打印找到的对应action
                                                # print(
                                                #     f"找到匹配的memory_id: {memory}，对应的action为: {buffer.actions[index_tmp]}")
                                                #print(f"agents的个数:{len(self.agents)}")
                                                #add_value_temp = ( (1 / len(route)) * 10) * 10


                                                    #print(f"第一次:{buffer.actions[index_tmp][0][add_value_index_i][0]}")
                                                if len(route) > 2:
                                                    for route_index in range(0,len(route)-2):
                                                        add_value_temp = ((1 / (len(route)-route_index)) * 10) * 10
                                                        current_rewards = added_value[route[route_index]][route[route_index+1]]
                                                        added_value[route[route_index]][route[route_index+1]] =max(current_rewards,add_value_temp)
                                                        added_value_1_flag[route[route_index]][route[route_index+1]] = 1
                                                        if route_index == (len(route) - 3):
                                                            added_value[route[route_index]][
                                                                route[route_index + 1]] = 600
                                                # current_rewards =added_value[agent.buoy.id][int(buffer.actions[index_tmp][0][agent.buoy.id][0])]
                                                # added_value[agent.buoy.id][int(buffer.actions[index_tmp][0][agent.buoy.id][0])]=max(current_rewards,add_value_temp)
                                                #print(f"agent.buoy_id:{agent.buoy.id},buffer.actions:{int(buffer.actions[index_tmp][0][agent.buoy.id][0])}")
                                                #print(f"added_value:{added_value}")

                                                # 原有的奖励更新逻辑
                                                # for i_t in range(len(buffer.rewards[index_tmp])):
                                                #     for j_t in range(len(buffer.rewards[index_tmp][i_t])):
                                                #         for k_t in range(len(buffer.rewards[index_tmp][i_t][j_t])):
                                                #             print("  ")
                                                            # buffer.rewards[index_tmp][i_t][j_t][k_t] += (r_q_time + (
                                                            #             1 / len(route)) * 10) * 10
                                                #break  # 找到后退出循环
                                        # for index_tmp, buffer_memory_id in enumerate(buffer.memory_list):
                                        #     if memory == buffer_memory_id:
                                        #         for i_t in range(len(buffer.rewards[index_tmp])):
                                        #             for j_t in range(len(buffer.rewards[index_tmp][i_t])):
                                        #                 for k_t in range(len(buffer.rewards[index_tmp][i_t][j_t])):
                                        #                     buffer.rewards[index_tmp][i_t][j_t][k_t] +=  (r_q_time + (1 / len(route)) * 10)*10


                                                #r = buffer[index][0][0][0]
                                                #r = r + (r_q_time + (1 / len(route)) * 10)*10
                                                #s_ = trans[3]
                                                #new_done = trans[4]
                                                #agent.RL.store_transition(s, a, r, s_, new_done, True)
                                                self.reward += ( (1 / len(route)) * 10)*10
                                                break
                                        packet.memories[i] = -1
                                        break

                                if packet.change_n == len(route) - 1:
                                    self.len.append(len(route))
                                    packet.change_n += 1

                                    to_del.append(index)
                        #print(len(packets))
                        #print(len(to_del))
                        for i in sorted(to_del, reverse=True):
                            #print("i",i)
                            del packets[i]
                        obs_n, agent_id_n, node_obs_n, adj_n, reward_n, done_n = self.env.step_one_agent(buoy.id, 0, False)
                        obs_list.append(obs_n[0])
                        agent_id_list.append(agent_id_n[0])
                        node_obs_list.append(node_obs_n[0])
                        adj_list.append(adj_n[0])
                        reward_list.append(reward_n[0])
                        done_list.append(done_n[0])
                        actions_mask_list.append([0])
                        continue
                    #print(buoy.id)
                    if buoy.if_sink:
                        obs_n, agent_id_n, node_obs_n, adj_n, reward_n, done_n = self.env.step_one_agent(buoy.id, 0, False)
                        obs_list.append(obs_n[0])
                        agent_id_list.append(agent_id_n[0])
                        node_obs_list.append(node_obs_n[0])
                        adj_list.append(adj_n[0])
                        reward_list.append(reward_n[0])
                        done_list.append(done_n[0])
                        actions_mask_list.append([0])
                        continue

                    observation = self.env.get_obs_list(buoy.id)

                    action_origin = int(actions[0][buoy.id][0])
                    action = [buoy.id, action_origin]
                    reward, done, reach = 0.0, False, False
                    pass_added_reward = (0.0 if action_origin == self.env.buoy_number
                                         else added_value[buoy.id][action_origin])
                    if action_origin == self.env.buoy_number or packet_num_list[buoy.id] == 0:
                        obs_n, agent_id_n, node_obs_n, adj_n, reward_n, done_n = self.env.step_one_agent(buoy.id,0, done)
                        actions_mask_list.append([0])

                    else:
                        agent.step += 1
                        queue_before = (buoy.packet, self.env.buoys[action_origin].packet)
                        reward, done, reach = self.env.buoy_go(action,pass_added_reward,penalty)
                        wm_runtime = getattr(self.agents[0].RL, "wm_runtime", None)
                        if wm_runtime is not None:
                            wm_runtime.record_attempt(buoy.id, action_origin, reach)
                        if self.data_exporter is not None:
                            self.data_exporter.attempt(
                                self.env, buoy.id, action_origin, service_packet_ids.get(buoy.id),
                                reach, reward, pass_added_reward, penalty, queue_before)

                        obs_n, agent_id_n, node_obs_n, adj_n, reward_n, done_n =  self.env.step_one_agent(buoy.id, reward, done)
                        actions_mask_list.append([1])



                    obs_list.append(obs_n[0])
                    agent_id_list.append(agent_id_n[0])
                    node_obs_list.append(node_obs_n[0])
                    adj_list.append(adj_n[0])
                    reward_list.append(reward_n[0])
                    done_list.append(done_n[0])




                    #print("reward don reach")
                    #print(reward, done, reach)


                    #observation_ = self.env.get_obs_list(buoy.id)



                    #memory_id = agent.RL.store_transition(observation, action_origin, reward, observation_, done, False)
                    self.reward += reward

                    # 奖励值回溯
                    to_del = []
                    k = 0
                    index = -1
                    for packet in packets:
                        index += 1

                        if_sink = packet.if_sink
                        if not if_sink:

                            if (packet.agent_id == action[0] and reach
                                    and packet.id == service_packet_ids.get(buoy.id)):
                                if k == 0:
                                    k = 1
                                    packet.transmit(action[1])
                                    if self.pm_runtime is not None and packet.if_sink:
                                        self.pm_runtime.delivered(packet.id)
                                    if self.data_exporter is not None:
                                        self.data_exporter.transmitted(packet)
                                    packet.queue_order = next_queue_order
                                    next_queue_order += 1
                                    forwarded_packet_ids.add(packet.id)
                                    packet.add_memory(self.memory_id)
                                    packet.stay_time = 1
                        if_sink = packet.if_sink
                        if if_sink:
                            buffer = self.agents[0].RL.buffer
                            route = packet.route
                            # print(f"route: {route}")
                            for i in range(len(route)):
                                if route[i] == agent.buoy.id:
                                    memory = packet.memories[i]
                                    memory_id_poll.append(memory)
                                    packet.change_n += 1
                                    r_q_time = -(packet.q_time[i+1] / sum(packet.q_time)) * 5

                                    if memory == tme_b.memory_id:
                                        reward_list.append(0)
                                        # 打印找到的对应action

                                        # print("buffer reward",buffer.rewards[0])
                                    for index_tmp, buffer_memory_id in enumerate(buffer.memory_list):
                                        if memory == buffer_memory_id:
                                            # print(
                                            #     f"找到匹配的memory_id: {memory}，对应的action为: {buffer.actions[index_tmp]}")
                                            # print(f"agents的个数:{len(self.agents)}")
                                            # add_value_temp = ( (1 / len(route)) * 10) * 10
                                            if len(route) > 2:
                                                for route_index in range(0, len(route) - 2):
                                                    add_value_temp = ((1 / (len(route) - route_index)) * 10) * 10
                                                    current_rewards = added_value[route[route_index]][
                                                        route[route_index + 1]]
                                                    added_value[route[route_index]][route[route_index + 1]] = max(
                                                        current_rewards, add_value_temp)
                                                    added_value_1_flag[route[route_index]][route[route_index + 1]] = 1
                                                    if route_index ==(len(route)-3):
                                                        added_value[route[route_index]][route[route_index+1]] = 600







                                            # current_rewards = added_value[agent.buoy.id][
                                            #     int(buffer.actions[index_tmp][0][agent.buoy.id][0])]
                                            # added_value[agent.buoy.id][
                                            #     int(buffer.actions[index_tmp][0][agent.buoy.id][0])] = max(
                                            #     current_rewards, add_value_temp)
                                            # print(
                                            #     f"agent.buoy_id:{agent.buoy.id},buffer.actions:{int(buffer.actions[index_tmp][0][agent.buoy.id][0])}")
                                            #print(f"******added_value_*******:{added_value}")
                                            # for i_t in range(len(buffer.rewards[index_tmp])):
                                            #     for j_t in range(len(buffer.rewards[index_tmp][i_t])):
                                            #         for k_t in range(len(buffer.rewards[index_tmp][i_t][j_t])):
                                            #             print(" ")#buffer.rewards[index_tmp][i_t][j_t][k_t] += (r_q_time + (1 / len(route)) * 10) * 10
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
 #                   if (agent.step > 200) and (agent.step % 5 == 0):

#                        agent.RL.learn()

                for idx_flag,row_added_value_flag in enumerate(added_value_1_flag):
                    for idx_flag_j,j_added_value_flag in enumerate(row_added_value_flag):
                        if j_added_value_flag==0:

                            if (added_value[idx_flag][idx_flag_j]+penalty) >0:
                                for agent in self.agents:
                                    if agent.buoy.id ==idx_flag:
                                        if idx_flag_j in agent.buoy.neighbor:
                                            added_value[idx_flag][idx_flag_j] += 0
                                            break

                            # else:
                            #     added_value[idx_flag][idx_flag_j] = 0

                # print(f"更新移动后的邻接矩阵如下：{self.env.buoy_net.adj_matrix}")



                # -------------------------- 格式化打印邻接矩阵 --------------------------
                # print("\n" + "=" * 50)  # 分隔线，让输出更醒目
                # print(f"移动后更新的一跳邻接矩阵（{self.env.buoy_net.buoys_number}x{self.env.buoy_net.buoys_number}）：")
                # print("=" * 50)
                # # 核心：用numpy格式化打印，设置对齐方式、小数位、宽度
                # np.set_printoptions(
                #     formatter={'float': '{:>6.2f}'.format},  # 浮点数右对齐，占6位，保留2位小数
                #     linewidth=100,  # 每行最大宽度（避免换行混乱，根据矩阵大小调整）
                #     suppress=True  # 抑制科学计数法（邻接矩阵值较小，避免1e-05这类显示）
                # )
                # # 打印邻接矩阵（若矩阵过大，可先切片打印部分，例：adj_matrix[:10, :10]打印前10行10列）
                # print(self.env.buoy_net.adj_matrix)
                # # 重置numpy打印格式（避免影响后续其他打印）
                # np.set_printoptions()
                # print("=" * 50 + "\n")




                for packet in self.env.packets:
                    if not packet.if_sink and packet.id not in forwarded_packet_ids:
                        packet.stay_time += 1
                self.step += 1
                self.env.current_step = self.step
                self.env.current_time_step = self.step
                if self.step % 20 == 0:
                    self.env.buoy_net.reset_lost_rate()
                if self.moving_buoy_ids:
                    for agent in self.agents:
                        if agent.buoy.id in self.moving_buoy_ids:
                            agent.buoy.move()
                    self.env.buoy_net.update_topology(added_value)
                # Capture t+1 only after all nodes and topology have advanced.
                obs_list, agent_id_list, node_obs_list, adj_list = self.env.observe_graph()
                available_tp1 = self.env.available_actions()
                terminal = self.step == self.env.all_args.episode_length
                done_batch = np.full((1, self.env.buoy_number), terminal, dtype=bool)

                reward_tmp = np.sum(reward_list)
                temp_reward += reward_tmp
                if self.pm_runtime is not None:
                    pm_row = self.pm_runtime.advance(self.env, self.step, reward_tmp, terminal)
                    self.pm_writer.writerow(pm_row)
                    reward_tmp = pm_row["reward_train"]
                if self.data_exporter is not None:
                    self.data_exporter.end_step(self.env, added_value, reward_tmp, terminal)
                #print(f"reward_tep: {reward_tmp}")

                reward_list = [[reward_tmp]] * len(self.agents)
                #print("all agent has done,  reward_list " , reward_list,len(self.agents))

                #数据组装与存储
                data = (
                    [obs_list],
                    [agent_id_list],
                    [node_obs_list],
                    [adj_list],
                    [agent_id_list],
                    [reward_list],
                    available_tp1,
                    done_batch,
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                    self.memory_id,
                    memory_id_poll
                )



                # insert data into buffer
                self.agents[0].RL.insert(data)
                if terminal:
                    if self.data_behavior == "tg-mappo" and not getattr(self.env.all_args, "tg_evaluate", False):
                        self.agents[0].RL.compute()
                        train_infos = self.agents[0].RL.train()
                    #print(self.step,self.env.all_args.episode_length)

                    break


            # 原代码：计算交付率、平均跳数、平均时延
            rate = len(self.len) / self.env.packet_id if self.env.packet_id != 0 else 0  # 避免除零错误
            loss_num = self.env.packet_id - len(self.len)
            average_hops = ( sum(self.len)  +  self.env.all_args.episode_length* loss_num ) / self.env.packet_id if self.env.packet_id else 0  # 避免空列表错误
            average_q_time = sum(self.q_time) / len(self.q_time) if self.q_time else 0

            # 新增：记录当前episode的指标
            self.episode_indices.append(episode)  # 记录episode编号
            self.episode_rates.append(rate)  # 记录交付率
            self.episode_hops.append(average_hops)  # 记录平均跳数
            self.metric_writer.write(
                episode, rate, average_hops, average_q_time
            )

            # 新增：输出日志（包含当前episode的关键指标）
            self.logger.info(
                f"Episode {episode} | "
                f"交付率: {rate:.4f} | "
                f"平均跳数: {average_hops:.2f} | "
                f"平均时延: {average_q_time:.2f} | "
                f"temp_reward: {temp_reward: 2f} |"
            )

            # 原代码：打印到控制台（可选，日志已包含）
            print(
                f'episode {episode} | rate {rate:.4f} | average_hops {average_hops:.2f} | average_q_time {average_q_time:.2f} | temp_reward: {temp_reward:.2f}')

        learner = self.agents[0].RL
        if self.data_behavior == "tg-mappo" and hasattr(learner, "save_policy"):
            learner.save_policy(self.run_dir)
        # 新增：所有episode结束后绘制图像
        self.plot_metrics()
        self.metric_writer.close()

    # # 新增：绘图函数
    # def plot_metrics(self):
    #     # 创建图像目录
    #     plot_dir = "training_plots"
    #     os.makedirs(plot_dir, exist_ok=True)
    #
    #     # 绘制交付率曲线
    #     plt.figure(figsize=(10, 5))
    #     plt.plot(self.episode_indices, self.episode_rates, label="交付率 (Rate)", color='blue')
    #     plt.xlabel("Episode")
    #     plt.ylabel("rate")
    #     plt.title("交付率随训练回合的变化")
    #     plt.grid(True)
    #     plt.legend()
    #     plt.savefig(os.path.join(plot_dir, "rate_vs_episode.png"))
    #     plt.close()
    #
    #     # 绘制平均跳数曲线
    #     plt.figure(figsize=(10, 5))
    #     plt.plot(self.episode_indices, self.episode_hops, label="平均跳数 (Average Hops)", color='red')
    #     plt.xlabel("Episode")
    #     plt.ylabel("average_hops")
    #     plt.title("平均跳数随训练回合的变化")
    #     plt.grid(True)
    #     plt.legend()
    #     plt.savefig(os.path.join(plot_dir, "average_hops_vs_episode.png"))
    #     plt.close()
    #
    #     self.logger.info(f"图像已保存至 {plot_dir} 目录")




    def plot_metrics(self):
        # 1. 生成时间戳（格式：年-月-日_时-分，避免特殊字符）
        timestamp = time.strftime("%Y-%m-%d_%H-%M", time.localtime())
        # 2. 创建带时间戳的图像目录（格式：training_plots_2025-10-20_14-30）
        plot_dir = os.path.join(self.run_dir, "plots", "TG_MAPPO")
        os.makedirs(plot_dir, exist_ok=True)

        # -------------------------- 绘制交付率曲线（英文版本）--------------------------
        plt.figure(figsize=(10, 5))
        plt.plot(self.episode_indices, self.episode_rates, label="Delivery Rate", color='blue')
        plt.xlabel("Episode")  # X轴：训练回合
        plt.ylabel("Delivery Rate")  # Y轴：交付率
        plt.title("Delivery Rate vs Training Episodes")  # 标题：交付率随训练回合变化
        plt.grid(True)
        plt.legend()
        # 保存路径：带时间戳的目录 + 明确的英文文件名
        rate_save_path = os.path.join(plot_dir, "delivery_rate_vs_episode.png")
        plt.savefig(rate_save_path)
        plt.close()

        # -------------------------- 绘制平均跳数曲线（英文版本）--------------------------
        plt.figure(figsize=(10, 5))
        plt.plot(self.episode_indices, self.episode_hops, label="Average Hops", color='red')
        plt.xlabel("Episode")  # X轴：训练回合
        plt.ylabel("Average Hops")  # Y轴：平均跳数
        plt.title("Average Hops vs Training Episodes")  # 标题：平均跳数随训练回合变化
        plt.grid(True)
        plt.legend()
        # 保存路径：带时间戳的目录 + 明确的英文文件名
        hops_save_path = os.path.join(plot_dir, "average_hops_vs_episode.png")
        plt.savefig(hops_save_path)
        plt.close()

        # 日志输出：明确告知保存路径（含时间戳）
        self.logger.info(f"Training plots saved to directory: {plot_dir}")
        self.logger.info(f"Delivery rate plot path: {rate_save_path}")
        self.logger.info(f"Average hops plot path: {hops_save_path}")




