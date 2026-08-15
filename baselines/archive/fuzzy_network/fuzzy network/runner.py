import random

import numpy as np
import time
import matplotlib.pyplot as plt

import torch
from maddpg import MADDPG
from replaybuffer import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

class Runner:
    def __init__(self, env, args):
        # 要不要用args呢
        self.time = None
        self.reward = []
        self.n_agents = args.n_agents
        self.n_obs = args.n_obs
        self.route = []
        self.env = env
        self.total_step = 0
        self.done = False
        self.replay_buffer = ReplayBuffer(50000)
        self.episode_length = 20
        # self.episode_length = 70
        self.min_size = 600
        self.learn_interval = 20
        self.batch_size = 512
        self.maddpg = MADDPG(args)
        self.epsilon = 1
        self.returns = []
        self.eva_interval = 5
        self.save_path = './result/model'
        self.route = []

    def train(self):
        for episode in range(50000):

            # 随机种子，因为数据包生成的时候，每次都搞成0了，所以每个episode都完全一样了，后面要考虑改一下
            # print('suijishu', np.random.normal(0.5, 100))

            obs_all = self.env.reset()

            nodes = self.env.nodes
            # print(nodes[2].packet.size())
            self.done = False
            returns = 0
            time_0 = time.time()


            # if episode != 0 and episode % self.eva_interval == 0:
            #     self.evaluate(obs_all, episode)
            #     obs_all = self.env.reset()

            # if episode == 1 :
            #     aaaa = obs_all


            for e_i in range(self.episode_length):


                # score_for_train, forward_id = self.maddpg.get_score_all(obs_all, self.epsilon,0)

                state = np.array(self.env.get_state())

                # print(state)



                score_for_train = self.maddpg.get_score_all(state, self.epsilon, 0)

                forward_id = self.maddpg.get_forward_id(score_for_train, nodes)



                # print(f_score)
                state = np.array(self.env.get_state())
                fuzzy_state = self.env.get_fuzzy_state()
                # print(fuzzy_state)
                # print('forward_id', forward_id)
                # print('score', score_for_train)
                reward, done = self.env.step(forward_id)

                next_state = np.array(self.env.get_state())
                next_fuzzy_state = self.env.get_fuzzy_state()

                sum_reward = np.sum(reward)

                returns += sum_reward



                # reward 也是列表吗
                self.replay_buffer.add(state, fuzzy_state, score_for_train, reward, next_state, next_fuzzy_state, done)

                # obs_all = self.env.get_obs_all()


                if self.replay_buffer.size(
                ) >= self.min_size and self.total_step % self.learn_interval == 0:

                    sample = self.replay_buffer.sample(self.batch_size)
                    def stack_array(x):
                        rearranged = [[sub_x[i] for sub_x in x]
                                      for i in range(len(x[0]))]
                        return [
                            torch.FloatTensor(np.vstack(aa)).to(device)
                            # torch.FloatTensor(np.vstack(aa))
                            for aa in rearranged
                        ]
                    sample_tensor = []
                    s_i = 0
                    for x in sample:
                        # if s_i == 0 or s_i == 4:
                        #     sample_tensor.append(torch.FloatTensor(x).to(device))
                        #     s_i += 1
                        #     continue

                        sample_tensor.append(stack_array(x))
                        s_i += 1



                    for a_i in range(self.n_agents):
                        # print('learn')
                        self.maddpg.learn(sample_tensor, a_i)

                    self.maddpg.update_all_targets()
                self.total_step += 1

            # agent = self.maddpg.agent
            # agent.scheduler_a.step()
            # agent.scheduler_c.step()
            # self.env.render(





            if self.epsilon >= 0.003:
                self.epsilon -= 0.0025
            else:self.epsilon = 0


            time_1 = time.time()
            r_time = time_1-time_0

            if episode % 10 == 0:
                self.maddpg.save_model(episode, self.save_path)

            dr = 0
            for packet in self.env.packets:
                if packet.if_sink:
                    dr += 1

            rate = dr/len(self.env.packets)
                # rate = 0

            print(f"第{episode}轮的return为: {returns} ")
            print(f"第{episode}轮的数据包到达率为: {rate} ")
            print(f"第{episode}轮的epsilon为: {self.epsilon} ")
            print(f"第{episode}轮的时间为: {r_time} ")
            self.returns.append(returns)
            with open("result/return.txt", "a") as f:
                print(returns, file=f)

            with open("result/dr.txt", "a") as f:
                print(rate, file=f)


    def test(self, packet_number):
        obs_all = self.env.reset_test(packet_number)
        v_e_all = []
        returns = 0
        nodes = self.env.nodes

        for e_i in range(self.episode_length):

            # pos = []
            # for node in self.env.nodes:
            #     pos.append(node.position)
            # with open("result/pos.txt", "a") as f:
            #     print(pos, file=f)

            state = np.array(self.env.get_state())

            score_for_train = self.maddpg.get_score_all(state,0, 0)

            forward_id = self.maddpg.get_forward_id(score_for_train, nodes)

            reward, done, v_e = self.env.step_test(forward_id)
            v_e_all.append(v_e)
            sum_reward = np.sum(reward)


            returns += sum_reward

            obs_all = self.env.get_obs_all()

            # break


            # # 记录位置
            # pos = []
            # for node in self.env.nodes:
            #     pos.append(node.position)
            # with open("result/pos.txt", "a") as f:
            #     print(pos, file=f)

        energy = []
        for node in self.env.nodes:
            energy.append(node.energy)

        dr = 0
        delay = 0
        for packet in self.env.packets:
            self.route.append(packet.route)
            delay += packet.dy
            if packet.if_sink:
                # delay += sum(packet.q_time)
                # delay += sum(packet.q_time)
                dr += 1
            # else:
            #     delay += 20

        rate = dr / len(self.env.packets)
        delay = delay / len(self.env.packets)
        print("rate", rate)
        print('delay', delay)
        print('return', returns)

        # x_values = list(range(len(v_e_all)))
        #
        # # 创建折线图
        # plt.plot(x_values, v_e_all, marker='o')  # marker='o' 表示在数据点上标记圆点
        #
        # # 添加标题和轴标签
        # # plt.title('折线图示例')
        # plt.xlabel('v')
        # plt.ylabel('step')
        #
        # # 显示图表
        # plt.show()


        # print(f"第{episode}轮的return为: {returns} ")
        # print(f"第{episode}轮的数据包到达率为: {rate} ")
        #w
        # with open("result/v_e_0.2.txt", "a") as f:
        #     print(v_e_all, file=f)
        #
        # with open("result/route.txt", "a") as f:
        # #     print(self.route, file=f)
        # with open("result/energy.txt", "a") as f:
        #     print(energy, file=f)

