import math
import random

import numpy as np
from .buoy import buoy_net, buoy_node, data_packet
import matplotlib.pyplot as plt
from .agent import Agent
from typing import Optional, Tuple, List
import argparse
import numpy as np
from numpy import ndarray as arr
from scipy import sparse
import os, sys
import gym
from gym import spaces
# random.seed(123)


class buoy_environment:
    def __init__(self, buoys,all_args):
        self.buoy_number = len(buoys)
        self.sink = None  # sink节点坐标
        self.sink_id = None
        self.buoys_info = buoys  # 创建对象的信息
        self.buoys = None  # 浮标对象
        self.buoy_net = None
        self.lambda_val = 2
        self.packets = []
        self.packet_id = 0
        self.current_step=0
        self.current_time_step = 0
        self.route_kong = True
        self.all_args=all_args
        self.step_number = 0
        self.cached_dist_vect = None
        # configure spaces
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = (
            []
        )  # adding this for compatibility with MAPPO code

     # adding this for compatibility with MAPPO code
    def _get_entity_feat_global(self, entity) -> arr:
        """
        Returns: ([velocity, position, goal_pos, entity_type])
        in global coords for the given entity
        """
        pos = entity.id
        #goal_pos = self.sink
        #entity_type = entity.if_sink


        return np.hstack([pos])

    def graph_observation(self, bouy: buoy_node) -> Tuple[arr, arr]:
        """
        FIXME: Take care of the case where edge_list is empty
        Returns: [node features, adjacency matrix]
        • Node features (num_entities, num_node_feats):
            If `global`:
                • node features are global [pos, vel, goal, entity-type]
                • edge features are relative distances (just magnitude)
                NOTE: for `landmarks` and `obstacles` the `goal` is
                        the same as its position
            If `relative`:
                • node features are relative [pos, vel, goal, entity-type] to ego agents
                • edge features are relative distances (just magnitude)
                NOTE: for `landmarks` and `obstacles` the `goal` is
                        the same as its position
        • Adjacency Matrix (num_entities, num_entities)
            NOTE: using the distance matrix, need to do some post-processing
            If `global`:
                • All close-by entities are connectd together
            If `relative`:
                • Only entities close to the ego-agent are connected

        """
        num_entities = self.buoy_number
        # node observations
        if self.cached_dist_vect is None:
            self.cached_dist_vect = np.zeros(
                (num_entities, num_entities, 2)
            )
        for ia, entity_a in enumerate(self.buoys):
            for ib in range(ia + 1, num_entities):
                entity_b = self.buoys[ib]
                delta_pos = [entity_a.position[0] - entity_b.position[0], entity_a.position[1] - entity_b.position[1]]
                delta_pos_i = [entity_b.position[0] - entity_a.position[0], entity_b.position[1] - entity_a.position[1]]
                self.cached_dist_vect[ia, ib, :] = [ia,ib]
                self.cached_dist_vect[ib, ia, :] = [ib,ia]

        self.cached_dist_mag = np.linalg.norm(self.cached_dist_vect, axis=2)


        node_obs = []
        for i, buoy in enumerate(self.buoys):
            node_obs_i = self._get_entity_feat_global(buoy)
            node_obs.append(node_obs_i)


        node_obs = np.array(node_obs)
        adj = self.cached_dist_mag

        return node_obs, adj

    def set_graph_obs_space(self):
        self.node_observation_space = []
        self.adj_observation_space = []
        self.edge_observation_space = []
        self.agent_id_observation_space = []
        self.share_agent_id_observation_space = []
        num_agents = self.buoy_number

        share_obs_dim = 0


        for bouy in self.buoys:
            node_obs, adj = self.graph_observation(bouy)
            node_obs_dim = node_obs.shape
            adj_dim = adj.shape
            edge_dim = 1  # NOTE hardcoding edge dimension
            agent_id_dim = 1  # NOTE hardcoding agent id dimension
            self.node_observation_space.append(
                spaces.Box(
                    low=-np.inf, high=+np.inf, shape=node_obs_dim, dtype=np.float32
                )
            )
            self.adj_observation_space.append(
                spaces.Box(low=-np.inf, high=+np.inf, shape=adj_dim, dtype=np.float32)
            )
            self.edge_observation_space.append(
                spaces.Box(
                    low=-np.inf, high=+np.inf, shape=(edge_dim,), dtype=np.float32
                )
            )
            self.agent_id_observation_space.append(
                spaces.Box(
                    low=-np.inf, high=+np.inf, shape=(agent_id_dim,), dtype=np.float32
                )
            )
            self.share_agent_id_observation_space.append(
                spaces.Box(
                    low=-np.inf,
                    high=+np.inf,
                    shape=(num_agents * agent_id_dim,),
                    dtype=np.float32,
                )
            )
            obs_dim = len(self.observation(bouy))

            share_obs_dim += obs_dim
            self.observation_space.append(
                spaces.Box(
                    low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32
                )
            )
            u_action_space = spaces.Discrete(self.buoy_number)
            self.action_space.append(u_action_space)
        self.share_observation_space = [
            spaces.Box(
                low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32
            )
            for _ in range(self.buoy_number)
        ]

    def _get_obs(self, agent: buoy_node) -> np.ndarray:
        return self.observation(agent=agent)

    def observation(self, agent: buoy_node) -> arr:
        """
        Return:
            [agent_vel, agent_pos, goal_pos]
        """
        # get positions of all entities in this agent's reference frame
        obs_list=[]
        obs_lost_rate = agent.success_rate_neighbor
        #obs_lost_rate.append(agent.packet)
        #for and_i in range(self.buoy_number):
        obs_list.append(agent.id)

        #print("obs_lost_rate",obs_list )

        goal_pos = []
        goal_packet =[]
        agents_goal = self.sink
        #goal_pos.append(    [agents_goal[0] - agent.position[0], agents_goal[0] - agent.position[0],agents_goal[0] - agent.position[0] ]     )

        goal_pos.append(obs_lost_rate)
        goal_packet.append(obs_list)


        #print("rate",goal_pos)
        #print("packet", goal_packet)


        #return  np.concatenate( [[1,2,3]] + goal_pos+goal_pos)



        return  np.concatenate(goal_pos+goal_packet)

    def build_env(self):
        self.step_number = 0
        self.buoy_net = buoy_net(self.buoys_info,self.all_args)

        self.buoys = self.buoy_net.get_buoys()
        self.sink_id = len(self.buoys) - 1
        buoy_sink = self.buoys[self.sink_id]
        self.sink = buoy_sink.position



        buoy_sink.if_sink = True
        buoy_sink.energy = 9999
        self.buoys.pop()
        self.buoys.append(buoy_sink)

        for buoy in self.buoys:
            for i in buoy_sink.neighbor:
                buoy.sink_neighbor.append(i)





        observations = []



        info = self.get_net_info()
        self.set_graph_obs_space()

        return observations, info

        # get observation for a particular agent  观察一个特定的代理


    def reset_graph(self) -> Tuple[List, List, List, List]:
        self.current_step = 0
        obs_n, node_obs_n, adj_n, agent_id_n = [], [], [], []
        self.agents = self.buoys
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            agent_id_n.append(np.array([agent.id]) )
            node_obs, adj = self.graph_observation(agent)
            node_obs_n.append(node_obs)
            adj_n.append(adj)
        #print(obs_n,agent_id_n,node_obs_n,adj_n)
        #obs_t = np.array([obs_n])
        #print("bos_n", obs_n)
        return obs_n, agent_id_n, node_obs_n, adj_n
    def reset(self):

        for buoy in self.buoys:
            buoy.node_refresh()
        self.packets = []
        self.packet_id = 0
        self.step_number = 0

    # 观测(智能体网络的输入)在强化学习当中好像只起到了加速收敛的作用
    # 这里需要考虑怎么记录数据包的路径
    def buoy_go(self, action, pass_added_reward,penalty=-35):
        #
        # if self.step_number % (2*self.buoy_number) == 0 and self.packet_id < (5*self.buoy_number):
        #
        #     self.packet_generation()
        # if self.step_number % 15 == 0:
        #     print("move", self.step_number/15)
        #     self.move()
        self.step_number += 1
        step_penalty = penalty
        pass_added_reward +=step_penalty
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
            is_adj = False
            # print(f"-----------------------------------------is_adj:{is_adj}")


        # print('is_adj',is_adj)
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


            reward = r_d + r_lost_rate+pass_added_reward





        else:

            reward = -0.5 +pass_added_reward

        # print(f"                                              action:{action},reward:{reward}")
        return reward, done, reach




    def step(self, action_n: List) -> Tuple[List, List, List, List, List, List, List]:
        self.current_step += 1
        obs_n, reward_n, done_n, info_n = [], [], [], []
        node_obs_n, adj_n, agent_id_n = [], [], []
        self.current_time_step += 1
        self.agents = self.buoys
        # set action for each agent
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent, self.action_space[i])
        # advance world state
        self.world.step()
        # record observation for each agent
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            agent_id_n.append(self.np.array([agent.id]))
            node_obs, adj = self._get_graph_obs(agent)
            node_obs_n.append(node_obs)
            adj_n.append(adj)
            reward = self._get_reward(agent)
            reward_n.append(reward)
            done_n.append(self._get_done(agent))
            info = {"individual_reward": reward}
            env_info = self._get_info(agent)
            info.update(env_info)  # nothing fancy here, just appending dict to dict
            info_n.append(info)
        # all agents get total reward in cooperative case
        reward = np.sum(reward_n)
        if self.shared_reward:
            reward_n = [[reward]] * self.n  # NOTE this line is similar to PPOEnv

        return obs_n, agent_id_n, node_obs_n, adj_n, reward_n, done_n, info_n




    def step_one_agent(self, agent_id: int, reward: int, done: bool) -> Tuple[List, List, List, List, List, List]:
        self.current_step += 1
        obs_n, reward_n, done_n, info_n = [], [], [], []
        node_obs_n, adj_n, agent_id_n = [], [], []
        self.current_time_step += 1
        self.agents = self.buoys
        # set action for each agent
        #buoy_go TODO

        agent = self.agents[agent_id]

        obs_n.append(self._get_obs(agent))
        agent_id_n.append(np.array([agent.id]))
        node_obs, adj = self.graph_observation(agent)
        node_obs_n.append(node_obs)
        adj_n.append(adj)
        reward_n.append(reward)
        done_n.append(done)



        # all agents get total reward in cooperative case
        #reward = np.sum(reward_n)
        #if self.shared_reward:
        #    reward_n = [[reward]] * self.n  # NOTE this line is similar to PPOEnv
        #
        return obs_n, agent_id_n, node_obs_n, adj_n, reward_n, done_n


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

            # random_number = np.random.poisson(lam=self.lambda_val)
            random_number = 1
            # if random_number == 0:
            #     random_number = 1
            buoy.packet += random_number
            for i in range(random_number):
                packet = data_packet(self.packet_id, buoy.id, self.sink_id)
                self.packets.append(packet)
                self.packet_id += 1

            # print(f"buoy_packet:{buoy.packet}")


    def packet_generation_one(self, number):
        buoy = self.buoys[0]
        buoy.packet += number
        for i in range(number):
            packet = data_packet(self.packet_id, buoy.id, self.sink_id)
            self.packets.append(packet)
            self.packet_id += 1


    def move(self):
        self.buoy_net.move()
