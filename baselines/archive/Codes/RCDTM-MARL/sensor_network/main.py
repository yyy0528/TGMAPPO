import random

from environment import buoy_environment
from runner import Runner
from agent import Agent

random.seed(12)


if __name__ == "__main__":

    # 浮标的格式[能力，坐标]，虽然能能量没有用到，但这个形式保留了下来
    buoys = [[100, [i, j, k]] for i in range(5) for j in range(5) for k in range(5)]

    env = buoy_environment(buoys)
    env.build_env()
    agent_number = env.buoy_number
    agents = []
    buoys_obj = env.buoys

    for i in range(agent_number):

        n_actions = env.get_n_action(i)
        n_features = env.get_n_obs(i)
        n_agents = env.buoy_number
        agent_tmp = Agent(buoys_obj[i], n_actions, n_features,n_agents)
        agents.append(agent_tmp)

    runner = Runner(agents, env)

    runner.run()