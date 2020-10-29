from __future__ import absolute_import, division, print_function
import argparse
import numpy as np
import torch
from utils.monitor import Monitor
from tensorboardX import SummaryWriter
from envs.mo_env import MultiObjectiveEnv

parser = argparse.ArgumentParser(description='MORL')
# CONFIG
parser.add_argument('--env-name', default='dst', metavar='ENVNAME',
                    help='environment to train on: dst | ft | ft5 | ft7')
parser.add_argument('--method', default='crl-naive', metavar='METHODS',
                    help='methods: crl-naive | crl-envelope | crl-energy')
parser.add_argument('--model', default='linear', metavar='MODELS',
                    help='linear | cnn | cnn + lstm')
parser.add_argument('--gamma', type=float, default=0.99, metavar='GAMMA',
                    help='gamma for infinite horizonal MDPs')
# TRAINING
parser.add_argument('--mem-size', type=int, default=4000, metavar='M',
                    help='max size of the replay memory')
parser.add_argument('--batch-size', type=int, default=256, metavar='B',
                    help='batch size')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate')
parser.add_argument('--epsilon', type=float, default=0.5, metavar='EPS',
                    help='epsilon greedy exploration')
parser.add_argument('--epsilon-decay', default=False, action='store_true',
                    help='linear epsilon decay to zero')
parser.add_argument('--weight-num', type=int, default=32, metavar='WN',
                    help='number of sampled weights per iteration')
parser.add_argument('--episode-num', type=int, default=2000, metavar='EN',
                    help='number of episodes for training')
parser.add_argument('--optimizer', default='Adam', metavar='OPT',
                    help='optimizer: Adam | RMSprop')
parser.add_argument('--update-freq', type=int, default=100, metavar='OPT',
                    help='optimizer: Adam | RMSprop')
parser.add_argument('--beta', type=float, default=0.01, metavar='BETA',
                    help='(initial) beta for evelope algorithm, default = 0.01')
parser.add_argument('--homotopy', default=False, action='store_true',
                    help='use homotopy optimization method')
# LOG & SAVING
parser.add_argument('--serialize', default=False, action='store_true',
                    help='serialize a model')
parser.add_argument('--save', default='crl/naive/saved/', metavar='SAVE',
                    help='path for saving trained models')
parser.add_argument('--name', default='', metavar='name',
                    help='specify a name for saving the model')
parser.add_argument('--log', default='crl/naive/logs/', metavar='LOG',
                    help='path for recording training informtion')

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


def train(env, agent, args):
    print(args.save)
    writer = SummaryWriter('./runs/exp-1')
    # monitor = Monitor(train=True, spec="-{}".format(args.method))
    # monitor.init_log(args.log, "m.{}_e.{}_n.{}".format(args.model, args.env_name, args.name))
    env.reset()
    for num_eps in range(args.episode_num):
        terminal = False
        env.reset()
        loss = 0
        cnt = 0
        tot_reward = 0

        probe = None
        if args.env_name == "dst":
            probe = FloatTensor([0.8, 0.2])
        elif args.env_name in ['ft', 'ft5', 'ft7']:
            # ft
            probe = FloatTensor([0.8, 0.2, 0.0, 0.0, 0.0, 0.0])

        while not terminal:
            # terminal happens when we reach the leaves on a tree. in which it happens on the 6th
            # loop if the depth is 6.
            state = env.observe()
            # its self.current_state = np.array([0, 0]) at first.
            # then we find the wq and find the action by finding the index corresponding to the 
            # maximum of the wq.
            action = agent.act(state)
            next_state, reward, terminal = env.step(action)
            # print(reward)
            # if args.log:
            #     monitor.add_log(state, action, reward, terminal, agent.w_kept)
            # here we memorize the reward, next state, state, action and whether or not 
            # it was terminal.
            # then inside the memorize function, a new random preference is assigned, q is computed again
            # using the state and next state and the new preferences. Then using the equation in page 21
            # of the paper, the probaility to choose this trajectory is computed. 
            # [but the reward was for the last action, now we have defind a new q and a new weight and using the old
            # reward. Why?]
            agent.memorize(state, action, next_state, reward, terminal)
            loss += agent.learn()
            if cnt > 100:
                terminal = True
                agent.reset()
            tot_reward = tot_reward + (probe.cpu().numpy().dot(reward)) * np.power(args.gamma, cnt)
            cnt = cnt + 1

        _, q = agent.predict(probe)

        if args.env_name == "dst":
            act_1 = q[0, 3]
            act_2 = q[0, 1]
        elif args.env_name in ['ft', 'ft5', 'ft7']:
            act_1 = q[0, 1]
            act_2 = q[0, 0]

        if args.method == "crl-naive":
            act_1 = act_1.data.cpu()
            act_2 = act_2.data.cpu()
        elif args.method == "crl-envelope":
            act_1 = probe.dot(act_1.data)
            act_2 = probe.dot(act_2.data)
        elif args.method == "crl-energy":
            act_1 = probe.dot(act_1.data)
            act_2 = probe.dot(act_2.data)
        print("end of eps %d with total reward (1) %0.2f, the Q is %0.2f | %0.2f; loss: %0.4f" % (
            num_eps,
            tot_reward,
            act_1,
            act_2,
            # q__max,
            loss / cnt))
        # monitor.update(num_eps,
        #                tot_reward,
        #                act_1,
        #                act_2,
        #                #    q__max,
        #                loss / cnt)
        print(num_eps)
        writer.add_scalar('loss', loss / cnt, num_eps)
        writer.add_scalar('act_1', act_1, num_eps)
        writer.add_scalar('act_2', act_2, num_eps)
    # if num_eps+1 % 100 == 0:
    # 	agent.save(args.save, args.model+args.name+"_tmp_{}".format(number))
    agent.save(args.save, "m.{}_e.{}_n.{}".format(args.model, args.env_name, args.name))



if __name__ == '__main__':
    args = parser.parse_args()
    # print(args.save)
    # setup the environment
    env = MultiObjectiveEnv(args.env_name)

    # get state / action / reward sizes
    state_size = len(env.state_spec)
    # self.state_spec = [['discrete', 1, [0, self.tree_depth]],
    #                       ['discrete', 1, [0, 2 ** self.tree_depth - 1]]]
    action_size = env.action_spec[2][1] - env.action_spec[2][0]
    # self.action_spec = ['discrete', 1, [0, 2]]
    reward_size = len(env.reward_spec)
    # self.reward_spec = [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]
    # print(state_size) : 2
    # print(action_size) : 2
    # print(reward_size) : 6

    # generate an agent for initial training
    agent = None
    if args.method == 'crl-naive':
        from crl.naive.meta import MetaAgent
        from crl.naive.models import get_new_model
    elif args.method == 'crl-envelope':
        from crl.envelope.meta import MetaAgent
        from crl.envelope.models import get_new_model
    elif args.method == 'crl-energy':
        from crl.energy.meta import MetaAgent
        from crl.energy.models import get_new_model

    if args.serialize:
        # if we want to continue running the last model.
        model = torch.load("{}{}.pkl".format(args.save,
                                             "m.{}_e.{}_n.{}".format(args.model, args.env_name, args.name)))
    else:
        model = get_new_model(args.model, state_size, action_size, reward_size)
        # model: linear
        # state size: 2
        # action size: 2
        # reward size: 6
    agent = MetaAgent(model, args, is_train=True)

    train(env, agent, args)
