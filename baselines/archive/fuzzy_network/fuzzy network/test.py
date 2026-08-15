from environment import Environment
from agent import Agent
from arguments import get_args
from runner import Runner
import numpy as np
import random
random.seed(2)
np.random.seed(42)

if __name__ == "__main__":
    nodes = [[100, [x * 100, y * 100, z * 100]] for x in range(4) for y in range(4) for z in range(4)]

    # print(nodes)

    nodes.append([100, [200, 200, 400]])

    # nodes = [[100, [x * 100, y * 100, z * 100]] for x in range(3) for y in range(3) for z in range(3)]
    #
    # # print(nodes)
    #
    # nodes.append([100, [100, 100, 300]])
    # nodes = [[20, [x, y, z]] for x in range(3) for y in range(3) for z in range(3)]
    # nodes.append([20, [1, 1, 3]])
    env = Environment(nodes)
    info = env.build_env()
    n_agents = env.n_node-1
    n_obs = env.get_n_obs()
    # n_state = (n_obs+1) * env.n_node
    # n_state = n_obs + env.n_node
    n_state = 2 * n_obs + 1 + 5
    nodes = env.nodes
    args = get_args()

    args.n_agents = n_agents
    args.n_obs = n_obs
    args.n_state = n_state
    # print(n_state)
    args.n_actions = 1
    runner = Runner(env, args)


    runner.test(8)
    # env.render()


