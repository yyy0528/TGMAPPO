from environment import Environment
from agent import Agent
from arguments import get_args
from runner import Runner
import random
random.seed(2)

if __name__ == "__main__":
    # nodes = [[5, [x, y, z]] for x in range(3) for y in range(3) for z in range(3)]

    nodes = [[100, [x * 100, y * 100, z * 100]] for x in range(3) for y in range(3) for z in range(3)]

    # print(nodes)

    nodes.append([100, [100, 100, 300]])
    # nodes.append([5, [1, 1, 3]])
    env = Environment(nodes)
    info = env.build_env()
    n_agents = env.n_node-1
    n_obs = env.get_n_obs()
    n_state = (n_obs+1) * env.n_node
    nodes = env.nodes
    args = get_args()

    args.n_agents = n_agents
    args.n_obs = n_obs
    args.n_state = n_state
    # print(n_state)
    args.n_actions = 1

    node = env.nodes[9]
    print(node.neighbor)

    for i in range(40):
        env.move()
        print(node.neighbor)

    env.render()