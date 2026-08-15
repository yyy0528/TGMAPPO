import numpy as np
import torch
import torch.nn.functional as F

from agent import Agent
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = torch.device("cuda")

class MADDPG:
    def __init__(self, args):

        # 不复用网络
        self.agents = []

        # 复用网络
        self.agent = Agent(args)

        self.gamma = args.gamma
        self.tau = args.tau
        self.n_node = args.n_agents + 1
        self.critic_criterion = torch.nn.MSELoss()
        self.device = device
        self.n_agents = args.n_agents
        self.n_obs = args.n_obs

        self.actor_pointer = 0
        self.policy_update_freq = 2
        self.policy_noise = 0.2
        self.noise_clip = 0.5


    @property
    def policies(self):
        return [agt.actor for agt in self.agents]

    @property
    def target_policies(self):
        return [agt.target_actor for agt in self.agents]

    def not_get_score_all(self, obs_all, epsilon, eva):
        score_for_train = np.zeros(self.n_node, dtype=np.float32)


        forward_id = []

        for a_id in range(self.n_agents):
            cur_obs = obs_all[a_id]


            if cur_obs == []:
                forward_id.append(-1)

            else:
                # 这里可能需要把scores改成numpy类型
                cur_obs = np.array(cur_obs, dtype=np.float32)

                scores = np.zeros(len(cur_obs), dtype=np.float32)

                for candidate_id in range(len(cur_obs)):
                    # scores[candidate_id] = (self.agents[a_id].get_score(cur_obs[candidate_id], epsilon))

                    nid = int(cur_obs[candidate_id][0])

                    score = self.agent.get_score(cur_obs[candidate_id], epsilon, eva)
                    score_for_train[nid] = score


                    # scores[candidate_id] = (self.agent.get_score(cur_obs[candidate_id], epsilon))
                    scores[candidate_id] = score


            # print('agent',a_id, 'score_id', f_score_id)

                forward_id.append(scores.argmax())
            # if a_id == 0:
            #
            #     print('智能体:', a_id, '分数：', scores)

        return score_for_train, forward_id

    def get_score_all(self, state, epsilon, eva):
        score_for_train = np.zeros(self.n_node, dtype=np.float32)

        for a_id in range(len(state)):
            cur_obs = np.array(state[a_id], dtype=np.float32)
            score = self.agent.get_score(cur_obs, epsilon, eva)
            score_for_train[a_id] = score

        return score_for_train

    def get_forward_id(self, score, nodes):
        forward_id = []
        for node in nodes:
            # tmp_score = []

            tmp_score = np.zeros(len(node.neighbor), dtype=np.float32)
            if node.neighbor == []:
                forward_id.append(-1)
                continue
            idx = 0
            for candidate_id in node.neighbor:
                tmp_score[idx] = score[candidate_id]
                idx += 1
            forward_id.append(tmp_score.argmax())

        return forward_id






    # 基本上就在这里了


    def learn(self, sample, i_agent):
        obs, fuzzy_state, act, rew, next_obs, next_fuzzy_state, done = sample

        cur_agent = self.agent  # 正确选择当前Agent

        # 更新Critic
        cur_agent.critic_optimizer.zero_grad()

        # 生成所有Agent的目标动作
        # 这里应该根据节点生成
        all_target_act = [cur_agent.target_actor(next_obs[i_agent])]


        target_critic_input = torch.cat((*[next_fuzzy_state[i_agent]], *[next_obs[i_agent]], *all_target_act), dim=1)
        target_critic_value = rew[i_agent].view(-1, 1) + self.gamma * cur_agent.target_critic(target_critic_input) * (
                    1 - done[i_agent].view(-1, 1))


        # 计算当前Critic值

        action = [act[i_agent]]
        critic_input = torch.cat((*[fuzzy_state[i_agent]], *[obs[i_agent]], *action), dim=1)
        # critic_input = torch.cat((*[obs[i_agent]], *action), dim=1)
        critic_value = cur_agent.critic(critic_input)


        critic_loss = self.critic_criterion(critic_value, target_critic_value.detach())
        critic_loss.backward()

        torch.nn.utils.clip_grad_norm_(cur_agent.critic.parameters(), 0.1)
        cur_agent.critic_optimizer.step()

        # 更新Actor
        cur_agent.actor_optimizer.zero_grad()
        cur_actor_out = cur_agent.actor(obs[i_agent])
        all_actor_acs = []
        all_actor_acs.append(cur_actor_out)

        vf_in = torch.cat((*[fuzzy_state[i_agent]], *[obs[i_agent]], *all_actor_acs), dim=1)
        # vf_in = torch.cat((*[obs[i_agent]], *all_actor_acs), dim=1)
        # vf_in = torch.cat((*[fuzzy_state[i_agent]], vf_in), dim=1)

        actor_loss = -cur_agent.critic(vf_in).mean()
        # actor_loss = - cur_actor_out.mean()
        # if i_agent == 1:
        #
        #     print(actor_loss)
        actor_loss += (cur_actor_out ** 2).mean() * 1e-3  # 正则化项

        actor_loss.backward()

        torch.nn.utils.clip_grad_norm_(cur_agent.actor.parameters(), 0.1)
        cur_agent.actor_optimizer.step()


    # def learn(self, sample, i_agent):
    #     obs, fuzzy_state, act, rew, next_obs, next_fuzzy_state, done = sample
    #
    #     cur_agent = self.agent  # 正确选择当前Agent
    #
    #     # 更新Critic
    #     cur_agent.critic_optimizer.zero_grad()
    #
    #     # 生成所有Agent的目标动作
    #     # 这里应该根据节点生成
    #     all_target_act = [cur_agent.target_actor(next_obs[i]) for i in range(self.n_node)]
    #     target_critic_input = torch.cat((*next_obs, *all_target_act), dim=1)
    #     # target_critic_input = torch.cat((*next_fuzzy_state, *all_target_act), dim=1)
    #     target_critic_value = rew[i_agent].view(-1, 1) + self.gamma * cur_agent.target_critic(target_critic_input) * (
    #                 1 - done[i_agent].view(-1, 1))
    #
    #     # 计算当前Critic值
    #     critic_input = torch.cat((*obs, *act), dim=1)
    #     # critic_input = torch.cat((*fuzzy_state, *act), dim=1)
    #     critic_value = cur_agent.critic(critic_input)
    #     critic_loss = self.critic_criterion(critic_value, target_critic_value.detach())
    #     critic_loss.backward()
    #     cur_agent.critic_optimizer.step()
    #
    #     # 更新Actor
    #     cur_agent.actor_optimizer.zero_grad()
    #     cur_actor_out = cur_agent.actor(obs[i_agent])
    #     all_actor_acs = []
    #     for i, _obs in enumerate(obs):
    #         if i == i_agent:
    #             all_actor_acs.append(cur_actor_out)
    #         else:
    #             # 使用其他Agent的当前Actor并detach
    #             other_action = cur_agent.actor(_obs).detach()
    #             all_actor_acs.append(other_action)
    #     vf_in = torch.cat((*obs, *all_actor_acs), dim=1)
    #     # vf_in = torch.cat((*fuzzy_state, *all_actor_acs), dim=1)
    #
    #     actor_loss = -cur_agent.critic(vf_in).mean()
    #     actor_loss += (cur_actor_out ** 2).mean() * 1e-3  # 正则化项
    #
    #     actor_loss.backward()
    #     cur_agent.actor_optimizer.step()

    def learn_td3(self, sample, i_agent):

        obs, fuzzy_state, act, rew, next_obs, next_fuzzy_state, done = sample

        cur_agent = self.agent  # 正确选择当前Agent

        # 更新Critic
        cur_agent.critic_td3_optimizer.zero_grad()

        # 生成所有Agent的目标动作
        # 这里应该根据节点生成
        # all_target_act = [cur_agent.target_actor(next_obs[i]) for i in range(self.n_node)]
        # all_target_act = [cur_agent.target_actor(next_obs[i_agent])]
        all_target_act = []

        batch_a_next = cur_agent.target_actor(next_obs[i_agent])
        noise = (torch.randn_like(batch_a_next) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
        batch_a_next = (batch_a_next + noise).clamp(-1, 1)
        all_target_act.append(batch_a_next)

        # target_critic_input = torch.cat((*next_obs, *all_target_act), dim=1)

        target_critic_input = torch.cat((*[next_fuzzy_state[i_agent]], *[next_obs[i_agent]], *all_target_act), dim=1)

        Q1_next, Q2_next = cur_agent.target_critic_td3(target_critic_input)

        target_Q = rew[i_agent].view(-1, 1) + self.gamma * torch.min(Q1_next, Q2_next) * (
                1 - done[i_agent].view(-1, 1))

        # 计算当前Critic值
        # critic_input = torch.cat((*obs, *act), dim=1)
        # critic_value = cur_agent.critic(critic_input)
        action = [act[i_agent]]
        critic_input = torch.cat((*[fuzzy_state[i_agent]], *[obs[i_agent]], *action), dim=1)
        current_Q1, current_Q2 = cur_agent.critic_td3(critic_input)

        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        # critic_loss = self.critic_criterion(critic_value, target_critic_value.detach())
        critic_loss.backward()
        cur_agent.critic_optimizer.step()

        # if self.actor_pointer % self.policy_update_freq == 0:
        #     # 更新Actor
        #     cur_agent.actor_optimizer.zero_grad()
        #     cur_actor_out = cur_agent.actor(obs[i_agent])
        #     all_actor_acs = []
        #     for i, _obs in enumerate(obs):
        #         if i == i_agent:
        #             all_actor_acs.append(cur_actor_out)
        #         else:
        #             # 使用其他Agent的当前Actor并detach
        #             other_action = cur_agent.actor(_obs).detach()
        #             all_actor_acs.append(other_action)

        cur_agent.actor_optimizer.zero_grad()
        cur_actor_out = cur_agent.actor(obs[i_agent])
        all_actor_acs = []
        all_actor_acs.append(cur_actor_out)

        vf_in = torch.cat((*[fuzzy_state[i_agent]], *[obs[i_agent]], *all_actor_acs), dim=1)
            # vf_in = torch.cat((*obs, *all_actor_acs), dim=1)
        actor_loss = -cur_agent.critic_td3.Q1(vf_in).mean()
        actor_loss += (cur_actor_out ** 2).mean() * 1e-3  # 正则化项

        actor_loss.backward()
        cur_agent.actor_optimizer.step()

        self.update_all_targets()


    def learn_ddpg(self, sample, i_agent):

        obs, fuzzy_state, act, rew, next_obs, next_fuzzy_state, done = sample

        cur_agent = self.agent  # 正确选择当前Agent

        # 更新Critic
        cur_agent.critic_optimizer.zero_grad()

        # 生成所有Agent的目标动作

        all_target_act = [cur_agent.target_actor(next_obs[i_agent])]

        target_critic_input = torch.cat((*[next_obs[i_agent]], *all_target_act), dim=1)

        target_critic_value = rew[i_agent].view(-1, 1) + self.gamma * cur_agent.target_critic(target_critic_input) * (
                1 - done[i_agent].view(-1, 1))

        # 计算当前Critic值
        action = [act[i_agent]]
        critic_input = torch.cat((*[obs[i_agent]], *action), dim=1)
        critic_value = cur_agent.critic(critic_input)

        critic_loss = self.critic_criterion(critic_value, target_critic_value.detach())
        critic_loss.backward()
        cur_agent.critic_optimizer.step()

        # 更新Actor
        cur_agent.actor_optimizer.zero_grad()
        cur_actor_out = cur_agent.actor(obs[i_agent])
        all_actor_acs = []
        all_actor_acs.append(cur_actor_out)
        vf_in = torch.cat((*[obs[i_agent]], *all_actor_acs), dim=1)


        actor_loss = -cur_agent.critic(vf_in).mean()

        actor_loss += (cur_actor_out ** 2).mean() * 1e-3  # 正则化项

        actor_loss.backward()
        cur_agent.actor_optimizer.step()


    def learn_old(self, sample, i_agent):
        obs, fuzzy_state, act, rew, next_obs, next_fuzzy_state, done = sample

        cur_agent = self.agent  # 正确选择当前Agent

        # 更新Critic
        cur_agent.critic_optimizer.zero_grad()

        # 生成所有Agent的目标动作
        # 这里应该根据节点生成
        all_target_act = [cur_agent.target_actor(next_obs[i]) for i in range(self.n_node)]
        target_critic_input = torch.cat((*next_obs, *all_target_act), dim=1)
        # target_critic_input = torch.cat((*next_fuzzy_state, *all_target_act), dim=1)
        target_critic_value = rew[i_agent].view(-1, 1) + self.gamma * cur_agent.target_critic(target_critic_input) * (
                1 - done[i_agent].view(-1, 1))

        # 计算当前Critic值
        critic_input = torch.cat((*obs, *act), dim=1)
        # critic_input = torch.cat((*fuzzy_state, *act), dim=1)
        critic_value = cur_agent.critic(critic_input)
        critic_loss = self.critic_criterion(critic_value, target_critic_value.detach())
        critic_loss.backward()
        cur_agent.critic_optimizer.step()

        # 更新Actor
        cur_agent.actor_optimizer.zero_grad()
        cur_actor_out = cur_agent.actor(obs[i_agent])
        all_actor_acs = []
        for i, _obs in enumerate(obs):
            if i == i_agent:
                all_actor_acs.append(cur_actor_out)
            else:
                # 使用其他Agent的当前Actor并detach
                other_action = cur_agent.actor(_obs).detach()
                all_actor_acs.append(other_action)
        vf_in = torch.cat((*obs, *all_actor_acs), dim=1)
        # vf_in = torch.cat((*fuzzy_state, *all_actor_acs), dim=1)

        actor_loss = -cur_agent.critic(vf_in).mean()
        actor_loss += (cur_actor_out ** 2).mean() * 1e-3  # 正则化项

        actor_loss.backward()
        cur_agent.actor_optimizer.step()



    def update_all_targets(self):
        # for agt in self.agents:
        #     agt.soft_update(agt.actor, agt.target_actor, self.tau)
        #     agt.soft_update(agt.critic, agt.target_critic, self.tau)

        self.agent.soft_update(self.agent.actor, self.agent.target_actor, self.tau)
        self.agent.soft_update(self.agent.critic, self.agent.target_critic, self.tau)

    def save_model(self, episode, save_path):
        network = self.agent.actor
        # torch.save(network.state_dict(), self.save_path + '{}.pth'.format(path) + '_{}.pth'.format(self.save_time))
        # torch.save(network.state_dict(), self.save_path + 'ep_{}.pth'.format(path))
        torch.save(network.state_dict(), save_path + '/{}.pth'.format(episode))

    def fuzzy_state(self):
        pass
