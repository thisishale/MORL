from __future__ import absolute_import, division, print_function
import argparse
import numpy as np
import torch
import torch.optim as optim
import os
import copy
# from utils.monitor import Monitor
from tensorboardX import SummaryWriter
from crl.envelope.models.linear import EnvelopeLinearCQN
from envs.mo_env import MultiObjectiveEnv
from collections import deque
from collections import namedtuple
import pickle

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
parser.add_argument('--loss_iter', type=int, default=10, metavar='EN',
                    help='number of loops in each episode for training')
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
parser.add_argument('--save-mean', default='crl/naive/mean/', metavar='SAVE_MEAN',
                    help='path for saving trained models')
parser.add_argument('--name', default='', metavar='name',
                    help='specify a name for saving the model')
parser.add_argument('--log', default='crl/naive/logs/', metavar='LOG',
                    help='path for recording training informtion')
parser.add_argument('--checkpoint', default='crl/naive/ckpt/', metavar='CHECKPOINT',
                    help='path for saving checkpoints')
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


def train(env, agent, init_ep_num, args):
    print(args.save)
    writer = SummaryWriter(args.log)
    # loss_list = []
    loss_list.append(np.inf)
    # monitor = Monitor(train=True, spec="-{}".format(args.method))
    # monitor.init_log(args.log, "m.{}_e.{}_n.{}".format(args.model, args.env_name, args.name))
    env.reset()
    # print('*'*100)
    # print('blah')
    # print(init_ep_num)
    for num_eps in range(init_ep_num+1, args.episode_num):
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
            temp, mean_vector, count, memroy_loss, last_memory_loss, trans_mem, priority_mem = agent.learn()
            
            loss += temp
            if cnt > 100:
                terminal = True
                agent.reset()
            tot_reward = tot_reward + (probe.cpu().numpy().dot(reward)) * np.power(args.gamma, cnt)
            cnt = cnt + 1
        if loss< min(loss_list) and not loss == 0:
            print(loss)
            print(num_eps)
            loss_list.append(loss)
            agent.save(args.checkpoint, "m.{}_e.{}_n.{}_ep.{}".format(args.model,args.env_name, args.name, num_eps), num_eps, trans_mem, priority_mem, loss_list)
            
        _, q = agent.predict(probe)
        # print(q.shape) #size(1,2)
        if args.env_name == "dst":
            act_1 = q[0, 3]
            act_2 = q[0, 1]
        elif args.env_name in ['ft', 'ft5', 'ft7']:
            act_1 = q[0, 1]
            act_2 = q[0, 0]
            # so act_1 is the q for the first action, act_1 is the q for the second action.

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
        if (num_eps+1) % 100 == 0:
            agent.save(args.save, "m.{}_e.{}_n.{}".format(args.model, args.env_name, args.name), num_eps, trans_mem, priority_mem, loss_list)
    # if num_eps+1 % 100 == 0:
    # 	agent.save(args.save, args.model+args.name+"_tmp_{}".format(number))
    agent.save(args.save, "m.{}_e.{}_n.{}".format(args.model, args.env_name, args.name), num_eps, trans_mem, priority_mem, loss_list)
    np.savetxt(os.path.normpath(os.path.join(args.save_mean,args.name+'.txt')),mean_vector)



if __name__ == '__main__':
    print('blahblahblahblah')
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
        # model = get_new_model(args.model, state_size, action_size, reward_size)
        model = EnvelopeLinearCQN(state_size, action_size, reward_size)
        ckpt = torch.load("{}{}.pkl".format(args.save,
                                             "m.{}_e.{}_n.{}".format(args.model, args.env_name, args.name)))
        model.load_state_dict(ckpt['state_dict'])
        model_ = model
        model = copy.deepcopy(model)
        device = torch.device('cuda:0')
        model = model.to(device)
        model_ = model_.to(device)
        if args.optimizer == 'Adam':
            optimizer = optim.Adam(model_.parameters(), lr=args.lr)
        elif args.optimizer == 'RMSprop':
            optimizer = optim.RMSprop(model_.parameters(), lr=args.lr)
        optimizer.load_state_dict(ckpt['optimizer'])
        # optimizer = optimizer.cuda()
        count = ckpt['count']
        loss_memory = ckpt['loss_memory']
        last_loss_memory = ckpt['last_loss_memory']
        init_eps = ckpt['ep']
        # loss_list = ckpt['loss_list']
        loss_list = []
        trans_mem = deque()
        priority_mem = deque()
        trans = namedtuple('trans', ['s', 'a', 's_', 'r', 'd'])
        with open(os.path.normpath(os.path.join(args.save,args.name+'_trans_mem.txt')), 'rb') as f:
            x = pickle.load(f)
        trans_mem = deque(x)
        # for tuple_to_dic in x:
        #     # print(tuple_to_dic)
        #     # print(tuple_to_dic[0])
        #     trans_mem.append(tuple_to_dic)

        with open(os.path.normpath(os.path.join(args.save,args.name+'_priority_mem.txt')), 'rb') as f:
             priority_mem = pickle.load(f)
        # for tuple_to_dic in x:
        #     priority_mem.append(trans(**tuple_to_dic)) 
        # trans_mem = ckpt['trans_mem']
        # priority_mem = ckpt['priority_mem']
        mean = ckpt['mean_']
        # model = torch.load("{}{}.pkl".format(args.save,
        #                                      "m.{}_e.{}_n.{}".format(args.model, args.env_name, args.name)))
    else:
        loss_list = []
        model = get_new_model(args.model, state_size, action_size, reward_size)
        model_ = model
        model = copy.deepcopy(model)
        if args.optimizer == 'Adam':
            optimizer = optim.Adam(model_.parameters(), lr=args.lr)
        elif args.optimizer == 'RMSprop':
            optimizer = optim.RMSprop(model_.parameters(), lr=args.lr)
        count = 0
        loss_memory = np.zeros((args.loss_iter, args.batch_size*args.weight_num, 6))
        last_loss_memory = np.zeros((args.loss_iter, args.batch_size*args.weight_num, 6))
        init_eps = 0
        trans = namedtuple('trans', ['s', 'a', 's_', 'r', 'd'])
        trans_mem = deque()
        priority_mem = deque()
        mean = np.zeros((6,1))
        # model: linear
        # state size: 2
        # action size: 2
        # reward size: 6
    agent = MetaAgent(model, model_, optimizer, count, loss_memory, last_loss_memory, args, mean, trans_mem, priority_mem, trans, is_train=True)

    train(env, agent, init_eps, args)
    print('*'*100)
    print('blah')