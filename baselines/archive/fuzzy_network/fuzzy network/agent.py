import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fuzzy import fuzzy_state, fuzzy_score
from torch.optim.lr_scheduler import StepLR
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")


class Agent:
    def __init__(self, args):

        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        # state 是不是obs直接相加的呢，这个感觉得看看论文,要不要包括其他智能体的动作
        self.state_shape = args.n_state
        self.n_obs = args.n_obs
        self.node_feat = (28, 4) # 28个节点，特征是位置能能量
        # self.policy = QMIX(args)
        self.args = args
        self.actor = Actor(self.n_obs, 1).to(device)

        # self.actor.load_state_dict((torch.load('C:/doctor/博士期间论文/new scalable/fuzzy network/result/model/14000.pth')))
        # self.actor.load_state_dict(
        #     (torch.load('C:/doctor/博士期间论文/new scalable/fuzzy network/result/model/3500.pth')))
        # self.actor.load_state_dict(
        #     (torch.load('C:/doctor/博士期间论文/Scalable routing/scr_huatu/muc_0.1/model/4000.pth')))

        # total_params = sum(p.numel() for p in self.actor.parameters())
        # print(f"模型总参数量: {total_params:,}")


        self.target_actor = Actor(self.n_obs, 1).to(device)

        self.critic = Critic(self.state_shape, self.n_agents).to(device)

        self.target_critic = Critic(self.state_shape, self.n_agents).to(device)

        self.critic_td3 = Critic_td3(self.state_shape, self.n_agents).to(device)

        self.target_critic_td3 = Critic_td3(self.state_shape, self.n_agents).to(device)

        # self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.lr, weight_decay=1e-5)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.lr)
        # self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.lr, weight_decay=1e-5)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.lr)
        self.critic_td3_optimizer = torch.optim.Adam(self.critic_td3.parameters(), lr=args.lr, weight_decay=1e-5)

        # self.scheduler_a = StepLR(self.actor_optimizer, step_size=50, gamma=0.5)
        # self.scheduler_c = StepLR(self.critic_optimizer, step_size=50, gamma=0.5)

    def get_score(self, obs, noise_std, eva):
        obs = torch.tensor(obs).to(device)

        # print(obs)
        score = self.actor(obs)
        score = score.detach().cpu().numpy()

        # score = score.detach().numpy()

        # if np.random.uniform() < epsilon and eva == 0:
        #
        #     score = score + np.random.uniform()
        # x = np.random.normal(0, noise_std)

        score = score + np.random.normal(0, noise_std)

        # print(score)
        # score = np.random.normal(0.5, 100)
        # print(x)
        # score = score + x
        return score

    def soft_update(self, net, target_net, tau):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_dim, 100)

        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 50)
        # self.fc4 = nn.Linear(100, 50)

        self.fc6 = nn.Linear(50, 50)
        # self.dropout = nn.Dropout(0.5)
        self.fc5 = nn.Linear(50, action_dim)

    def forward(self, x):
        # print(x)
        x = F.relu(self.fc1(x))
        # assert torch.isnan(x).sum() == 0, print(x, '1111111')
        x = F.relu(self.fc2(x))
        # assert torch.isnan(x).sum() == 0, print(x, '222222222')
        x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))

        x = F.relu(self.fc6(x))
        # assert torch.isnan(x).sum() == 0, print(x, '333333333')
        # x = F.sigmoid(self.fc5(x))
        x = self.fc5(x)

        # x = self.fc5(x)
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(state_dim, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, 50)
        self.fc5 = nn.Linear(50, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

# class Critic(nn.Module):
#     def __init__(self, state_dim, node_feat_dim, action_dim):
#         super(Critic, self).__init__()
#         self.gcn_encoder = GCNEncoder(node_feat_dim, 100)
#         self.fc1 = nn.Linear(state_dim + 100, 100)
#         self.fc2 = nn.Linear(100, 100)
#         self.fc3 = nn.Linear(50, 1)
#
#     def forward(self, graph, edge_index, state):
#
#         g_embedding = self.gcn_encoder(graph, edge_index)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = F.relu(self.fc4(x))
#         x = self.fc5(x)
#         return x


class GCNEncoder(nn.Module):
    """GCN编码器：生成融合邻居信息的节点嵌入"""
    def __init__(self, node_feat_dim, hidden_dim):
        super().__init__()
        # 第一层GCN：输入节点特征，输出隐藏层特征
        self.gcn1 = GCNConv(
            in_channels=node_feat_dim,
            out_channels=hidden_dim,
            add_self_loops=True  # 加入自环（包含自身特征）
        )
        # 第二层GCN：进一步聚合
        self.gcn2 = GCNConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            add_self_loops=True
        )
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        """
        输入：
            x: [num_nodes, node_feat_dim] 节点特征
            edge_index: [2, num_edges] 边索引（GCN内部会转为邻接矩阵）
        输出：
            node_emb: [num_nodes, hidden_dim] 节点嵌入（融合自身与邻居）
        """
        x = self.gcn1(x, edge_index)  # 第一层GCN
        x = self.relu(x)
        x = self.gcn2(x, edge_index)  # 第二层GCN
        x = self.relu(x)
        return x



class Critic_td3(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic_td3, self).__init__()

        self.fc1 = nn.Linear(state_dim, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear(50, 1)

        self.fc5 = nn.Linear(state_dim, 100)
        self.fc6 = nn.Linear(100, 100)
        self.fc7 = nn.Linear(100, 50)
        self.fc8 = nn.Linear(50, 1)

    def forward(self, x):
        q1 = F.relu(self.fc1(x))
        q1 = F.relu(self.fc2(q1))
        q1 = F.relu(self.fc3(q1))
        q1 = self.fc4(q1)

        q2 = F.relu(self.fc1(x))
        q2 = F.relu(self.fc2(q2))
        q2 = F.relu(self.fc3(q2))
        q2 = self.fc4(q2)

        return q1, q2

    def Q1(self, x):
        q1 = F.relu(self.fc1(x))
        q1 = F.relu(self.fc2(q1))
        q1 = F.relu(self.fc3(q1))
        q1 = self.fc4(q1)

        return q1
