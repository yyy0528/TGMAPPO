import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--replay_dir', type=str, default=r'', help='absolute path to save the replay')
    parser.add_argument('--alg', type=str, default='qmix', help='the algorithm to train the agent')
    parser.add_argument('--last_action', type=bool, default=True, help='whether to use the last action to choose action')
    parser.add_argument('--reuse_network', type=bool, default=True, help='whether to use one network for all agents')
    parser.add_argument('--gamma', type=float, default=0.95, help='discount factor')
    parser.add_argument('--device', type=str, default='cuda', help='training device')


    parser.add_argument('--evaluate_epoch', type=int, default=20, help='number of the epoch to evaluate the agent')

    # 以下几个参数比较常用,注意只有指定变量名的时候，就会变成true，没法b变false，最好直接在这里改参数吧
    parser.add_argument('--model_dir', type=str, default='./MARL/model', help='model directory of the policy')
    parser.add_argument('--result_dir', type=str, default='./result', help='result directory of the policy')

    parser.add_argument('--load_model', type=bool, default=False, help='whether to load the pretrained model')
    parser.add_argument('--learn', type=bool, default=True, help='whether to train the model')
    parser.add_argument('--cuda', type=bool, default=True, help='whether to use the GPU')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--tau', type=float, default=0.01, help='some useful rate')

    args = parser.parse_args()
    return args

