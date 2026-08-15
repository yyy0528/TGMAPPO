import numpy as np
import random


def fuzzy_state(obs):
    '''
    处理二维数组[n_candidate, n_obs]
    :param obs:
    :return:
    '''
    random_integer = random.randint(0, len(obs)-1)
    return obs[random_integer], random_integer
    # return np.mean(obs, axis=0)


def fuzzy_score(scores, index):
    # return np.mean(scores)
    # random_integer = random.randint(0, len(scores)-1)
    return scores[index]
