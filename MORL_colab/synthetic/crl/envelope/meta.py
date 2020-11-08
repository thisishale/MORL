from __future__ import absolute_import, division, print_function
import random
import torch
import copy
import os
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from collections import namedtuple
from collections import deque
import pickle

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


class MetaAgent(object):
    '''
    (1) act: how to sample an action to examine the learning
        outcomes or explore the environment;
    (2) memorize: how to store observed observations in order to
        help learing or establishing the empirical model of the
        enviroment;
    (3) learn: how the agent learns from the observations via
        explicitor implicit inference, how to optimize the policy
        model.
    '''

    def __init__(self, model, model_, optimizer, count, memory_loss, last_memory_loss, args, mean, trans_mem, priority_mem, trans, is_train=False):
        # self.model_ = model
        # self.model = copy.deepcopy(model)
        self.mean_ = mean
        self.trans_mem = trans_mem
        self.priority_mem = priority_mem
        self.count = count
        self.loss_memory = memory_loss
        self.last_loss_memory = last_memory_loss
        self.model_ = model_
        self.model = model
        self.is_train = is_train
        self.gamma = args.gamma
        # 0.99
        self.epsilon = args.epsilon
        # 0.5
        self.epsilon_decay = args.epsilon_decay
        # true
        self.epsilon_delta = (args.epsilon - 0.05) / args.episode_num
        self.trans = trans
        self.mem_size = args.mem_size
        # 4000
        self.batch_size = args.batch_size
        # 256
        self.weight_num = args.weight_num
        # 32

        self.beta            = args.beta
        # 0.01
        self.beta_init       = args.beta
        self.beta_loss       = -0.1
        self.homotopy        = args.homotopy
        # true
        self.beta_uplim      = 1.00
        self.tau             = 1000.
        self.beta_expbase    = float(np.power(self.tau*(self.beta_uplim-self.beta), 1./args.episode_num))
        self.beta_delta      = self.beta_expbase / self.tau
        # self.trans_mem = deque()
        # self.trans = namedtuple('trans', ['s', 'a', 's_', 'r', 'd'])
        # self.priority_mem = deque()
        self.loss_iter = args.loss_iter
        # self.loss_memory = np.zeros((self.loss_iter, self.batch_size*self.weight_num, 6))
        # self.last_loss_memory = np.zeros((self.loss_iter, self.batch_size*self.weight_num, 6))
        # print('*'*100)
        # print(self.loss_memory.shape)
        # self.mean_ = np.zeros((6,1))
        self.cov_ = np.ones((6,1))
        self.sum_vec = 0
        self.optimizer = optimizer
        # if args.optimizer == 'Adam':
        #     self.optimizer = optim.Adam(self.model_.parameters(), lr=args.lr)
        # elif args.optimizer == 'RMSprop':
        #     self.optimizer = optim.RMSprop(self.model_.parameters(), lr=args.lr)

        self.w_kept = None
        self.update_count = 0
        self.update_freq = args.update_freq
        # 100

        if self.is_train:
            self.model.train()
        if use_cuda:
            self.model.cuda()
            self.model_.cuda()

    def act(self, state, preference=None):
        # random pick a preference if it is not specified
        if preference is None:
            if self.w_kept is None:
                # define a random weight in shape of rewards which is 6.
                self.w_kept = torch.randn(self.model_.reward_size)
                # normalize the weight.
                self.w_kept = (torch.abs(self.w_kept) / \
                               torch.norm(self.w_kept, p=1)).type(FloatTensor)
            preference = self.w_kept
        state = torch.from_numpy(state).type(FloatTensor)
        # Creates a Tensor from a numpy.ndarray.
        _, Q = self.model_(
            Variable(state.unsqueeze(0)),
            # (1,2)
            Variable(preference.unsqueeze(0)))
            # (1,6)
        # print(Q.shape) [1,2,6] 
        Q = Q.view(-1, self.model_.reward_size)
        # print(Q.shape) [2,6] (action, preference)
        Q = torch.mv(Q.data, preference)
        # print(Q.shape) [2]
        action = Q.max(0)[1].cpu().numpy()
        #print(Q.max(0)) (tensor(0.0192), tensor(0)) #first one if the number of max q 
        # which is for action 0(the second element in tensor)
        action = int(action)
        # If there is not enough data for a batch or we are at the start of training:
        # choose randomly.
        if self.is_train and (len(self.trans_mem) < self.batch_size or \
                              torch.rand(1)[0] < self.epsilon):
            action = np.random.choice(self.model_.action_size, 1)[0]
            action = int(action)

        return action

    def memorize(self, state, action, next_state, reward, terminal):
        self.trans_mem.append(self.trans(
            torch.from_numpy(state).type(FloatTensor),  # state
            action,  # action
            torch.from_numpy(next_state).type(FloatTensor),  # next state
            torch.from_numpy(reward).type(FloatTensor),  # reward
            terminal))  # terminal
        # print(len(self.trans_mem))
        # randomly produce a preference for calculating priority
        # preference = self.w_kept
        preference = torch.randn(self.model_.reward_size)
        preference = (torch.abs(preference) / torch.norm(preference, p=1)).type(FloatTensor)
        state = torch.from_numpy(state).type(FloatTensor)
        # preference size: [6]
        # state size: [2]
        # reward size: [6]
        _, q = self.model_(Variable(state.unsqueeze(0), requires_grad=False),
                           Variable(preference.unsqueeze(0), requires_grad=False))
        # q size: [1, 2, 6]
        q = q[0, action].data
        # chooses the row corresponding to that action q size: [6]
        wq = preference.dot(q)
        # this is wq(s,a,w) in page 21.
        # wq is a number now.
        wr = preference.dot(torch.from_numpy(reward).type(FloatTensor))
        # wr is a number now.
        if not terminal:
            next_state = torch.from_numpy(next_state).type(FloatTensor)
            # next_state : [2]
            # next_state.unsqueeze(0) : [1, 2]
            # preference.unsqueeze(0) : [1, 6]
            hq, _ = self.model_(Variable(next_state.unsqueeze(0), requires_grad=False),
                                Variable(preference.unsqueeze(0), requires_grad=False))
            # hq gives the row of q in which the action is max (after implementing w). 
            # hq id q_target in formula in paper page 21.
            # q gives the whole q. 
            hq = hq.data[0]
            whq = preference.dot(hq)
            # whq is a number now.
            p = abs(wr + self.gamma * whq - wq)
        else:
            print(self.beta)
            self.w_kept = None
            if self.epsilon_decay:
                self.epsilon -= self.epsilon_delta
            if self.homotopy:
                self.beta += self.beta_delta
                self.beta_delta = (self.beta-self.beta_init)*self.beta_expbase+self.beta_init-self.beta
            p = abs(wr - wq)
        p += 1e-5

        self.priority_mem.append(
            p
        )
        if len(self.trans_mem) > self.mem_size:
            self.trans_mem.popleft()
            self.priority_mem.popleft()

    def sample(self, pop, pri, k):
        pri = np.array(pri).astype(np.float)
        inds = np.random.choice(
            range(len(pop)), k,
            replace=False,
            p=pri / pri.sum()
        )
        # print(inds.shape) (256,)
        return [pop[i] for i in inds]

    def actmsk(self, num_dim, index):
        mask = ByteTensor(num_dim).zero_()
        mask[index] = 1
        return mask.unsqueeze(0)

    def nontmlinds(self, terminal_batch):
        mask = ByteTensor(terminal_batch)
        inds = torch.arange(0, len(terminal_batch)).type(LongTensor)
        inds = inds[mask.eq(0)]
        return inds

    def learn(self):
        # print(len(self.trans_mem))
        self.update_count += 1
        if len(self.trans_mem) > self.batch_size:
            # print('*'*50)
            # print(len(self.trans_mem))
            # print('*'*50)
            # print('I entered')
            for l_iter in range(self.loss_iter):
                if l_iter == self.loss_iter-1:
                    self.last_loss_memory = copy.deepcopy(self.loss_memory)
                    tmpmu = np.mean(np.mean(self.last_loss_memory, axis=0),axis=0)
                    # print(tmpmu.shape)
                    for i in range(len(tmpmu)):
                        self.sum_vec += np.exp(self.beta_loss*tmpmu[i])
                    self.mean_ = np.exp(self.beta_loss*tmpmu)/self.sum_vec
                # target model is 
                action_size = self.model_.action_size
                reward_size = self.model_.reward_size
                # print(action_size) 2
                # print(reward_size) 6
                minibatch = self.sample(self.trans_mem, self.priority_mem, self.batch_size)
                # print(len(minibatch)) 256
                # minibatch shape: (256,5,...(depends on what it is(state, next state, reward...)))
                # print(len(minibatch[0])) 5 
                # print(len(minibatch[0][0])) 2 state 
                # print(len(minibatch[0][1])) 0 int action
                # print(len(minibatch[0][2])) 2 next state
                # print(len(minibatch[0][3])) 6 reward
                # print(len(minibatch[0][4])) 0 bool  terminal or not
                # minibatch = random.sample(self.trans_mem, self.batch_size)
                batchify = lambda x: list(x) * self.weight_num
                state_batch = batchify(map(lambda x: x.s.unsqueeze(0), minibatch))
                # print(len(state_batch)) 8192
                # print(minibatch[0].s.shape) 2
                # print(minibatch[0].s.unsqueeze(0).shape) (1,2)
                # len(list(map(lambda x: x.s.unsqueeze(0), minibatch))) 256
                # len(list(map(lambda x: x.s.unsqueeze(0), minibatch))[0]) 1
                # len(list(map(lambda x: x.s.unsqueeze(0), minibatch))[0]) 2
                # so it gets repeated since it gets multiplied by 32 which is the weight_num.
                # it gets repeated for them all.
                # a is int, convert it to longtensor.
                action_batch = batchify(map(lambda x: LongTensor([x.a]), minibatch)) 
                reward_batch = batchify(map(lambda x: x.r.unsqueeze(0), minibatch))
                next_state_batch = batchify(map(lambda x: x.s_.unsqueeze(0), minibatch))
                terminal_batch = batchify(map(lambda x: x.d, minibatch))
                # so what i understand from this code is that, we found actions using diffeent preferences, 
                # and we found trajectories, then found probability of choosing each trajectory using a 
                # different random weights. and now we are training the model using different random weights, 
                # not the weights the model was trained with.
                if self.count<self.loss_iter-1: 
                    w_batch = np.random.randn(self.weight_num, reward_size)
                    # print(w_batch.shape)
                else:
                    # w_batch = np.random.normal(self.mean_,self.cov_,)
                    w_batch = np.random.normal(self.mean_.squeeze(),self.cov_.squeeze(),(self.weight_num, reward_size))
                    # print(self.mean_)
                    # print(w_batch.shape)
                # print(w_batch.shape) 32, 6
                w_batch = np.abs(w_batch) / \
                        np.linalg.norm(w_batch, ord=1, axis=1, keepdims=True)
                # print(w_batch.shape) 32, 6
                w_batch = torch.from_numpy(w_batch.repeat(self.batch_size, axis=0)).type(FloatTensor)
                # print(w_batch.shape) 8192,6
                __, Q = self.model_(Variable(torch.cat(state_batch, dim=0)),
                                    Variable(w_batch), w_num=self.weight_num)
                # print(Q.shape) 8192, 2, 6
                # detach since we don't want gradients to propagate
                # HQ, _    = self.model_(Variable(torch.cat(next_state_batch, dim=0), volatile=True),
                # 					  Variable(w_batch, volatile=True), w_num=self.weight_num)
                _, DQ = self.model(Variable(torch.cat(next_state_batch, dim=0), requires_grad=False),
                                Variable(w_batch, requires_grad=False))
                # print(DQ.shape) 8192, 2, 6
                w_ext = w_batch.unsqueeze(2).repeat(1, action_size, 1)
                # print(w_ext.shape) 8192, 12, 1
                w_ext = w_ext.view(-1, self.model.reward_size)
                # print(w_ext.shape) 16384, 6
                _, tmpQ = self.model_(Variable(torch.cat(next_state_batch, dim=0), requires_grad=False),
                                    Variable(w_batch, requires_grad=False))
                # print(tmpQ.shape) [8192, 2, 6]
                tmpQ = tmpQ.view(-1, reward_size)
                # print(tmpQ.shape) 16384, 6
                # print(torch.bmm(w_ext.unsqueeze(1),
                # 			    tmpQ.data.unsqueeze(2)).view(-1, action_size))
                act = torch.bmm(Variable(w_ext.unsqueeze(1), requires_grad=False),
                                tmpQ.unsqueeze(2)).view(-1, action_size).max(1)[1]
                # act 8192
                # print(DQ.size(2)) 6
                HQ = DQ.gather(1, act.view(-1, 1, 1).expand(DQ.size(0), 1, DQ.size(2))).squeeze()
                # print(HQ.shape)
                # 8192, 6
                nontmlmask = self.nontmlinds(terminal_batch)
                with torch.no_grad():
                    Tau_Q = Variable(torch.zeros(self.batch_size * self.weight_num,
                                                reward_size).type(FloatTensor))
                    Tau_Q[nontmlmask] = self.gamma * HQ[nontmlmask]
                    # Tau_Q.volatile = False
                    Tau_Q += Variable(torch.cat(reward_batch, dim=0))

                actions = Variable(torch.cat(action_batch, dim=0))

                Q = Q.gather(1, actions.view(-1, 1, 1).expand(Q.size(0), 1, Q.size(2))
                            ).view(-1, reward_size)
                Tau_Q = Tau_Q.view(-1, reward_size)

                wQ = torch.bmm(Variable(w_batch.unsqueeze(1)),
                            Q.unsqueeze(2)).squeeze()

                wTQ = torch.bmm(Variable(w_batch.unsqueeze(1)),
                                Tau_Q.unsqueeze(2)).squeeze()

                # loss = F.mse_loss(Q.view(-1), Tau_Q.view(-1))
                # print((1-self.beta) * F.mse_loss(Q.view(-1), Tau_Q.view(-1)))
                # print(wTQ.view(-1))
                loss = self.beta * F.mse_loss(wQ.view(-1), wTQ.view(-1))
                loss += (1-self.beta) * F.mse_loss(Q.view(-1), Tau_Q.view(-1))
                # print(self.loss_memory.shape)
                self.loss_memory[l_iter,:,:] = (Q-Tau_Q).cpu().detach().numpy()
                self.optimizer.zero_grad()
                loss.backward()
                for param in self.model_.parameters():
                    param.grad.data.clamp_(-1, 1)
                self.optimizer.step()

                if (self.update_count+1) % self.update_freq == 0:
                    self.model.load_state_dict(self.model_.state_dict())
                self.count = self.count + 1
            
                
            return loss.data, self.mean_, self.count, self.loss_memory, self.last_loss_memory, self.trans_mem, self.priority_mem

 ###################################################################
 ###############################################################
        return 0.0 , self.mean_, 0, self.loss_memory, self.last_loss_memory, self.trans_mem, self.priority_mem
    def reset(self):
        self.w_kept = None
        if self.epsilon_decay:
            self.epsilon -= self.epsilon_delta
        if self.homotopy:
            self.beta += self.beta_delta
            self.beta_delta = (self.beta-self.beta_init)*self.beta_expbase+self.beta_init-self.beta

    def predict(self, probe):
        return self.model(Variable(FloatTensor([0, 0]).unsqueeze(0), requires_grad=False),
                          Variable(probe.unsqueeze(0), requires_grad=False))

    def save(self, save_path, model_name, episode_num, trans_mem, priority_mem, loss_list):
        a = []
        temp_mem = copy.deepcopy(trans_mem)
        for i in range(len(temp_mem)):
            a.append(temp_mem.popleft())
        # print(a)
        with open(os.path.normpath(os.path.join(save_path,model_name+'_trans_mem.txt')), 'wb') as f:
            pickle.dump(a, f)
            
        # b = []
        # temp_mem = copy.deepcopy(priority_mem)
        # for i in range(len(temp_mem)):
        #     b.append(temp_mem.popleft()._asdict())
        # print(a)
        with open(os.path.normpath(os.path.join(save_path,model_name+'_priority_mem.txt')), 'wb') as f:
            pickle.dump(priority_mem, f)
            
        state = {
            'ep': episode_num,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'count' : self.count, 
            'loss_memory' : self.loss_memory,
            'last_loss_memory' : self.last_loss_memory,
            # 'loss_list': loss_list,
            'mean_' : self.mean_
            # 'trans_mem' : a,
            # 'priority_mem' : list(self.priority_mem)
        }
        torch.save(state, "{}{}.pkl".format(save_path, model_name))
        # torch.save(self.model, "{}{}.pkl".format(save_path, model_name))

    def find_preference(
            self,
            w_batch,
            target_batch,
            pref_param):

        with torch.no_grad():
            w_batch = FloatTensor(w_batch)
            target_batch = FloatTensor(target_batch)

        # compute loss
        pref_param = FloatTensor(pref_param)
        pref_param.requires_grad = True
        sigmas = FloatTensor([0.001]*len(pref_param))
        dist = torch.distributions.normal.Normal(pref_param, sigmas)
        pref_loss = dist.log_prob(w_batch).sum(dim=1) * target_batch

        self.optimizer.zero_grad()
        # Total loss
        loss = pref_loss.mean()
        loss.backward()
        
        eta = 1e-3
        pref_param = pref_param + eta * pref_param.grad
        pref_param = simplex_proj(pref_param.detach().cpu().numpy())
        # print("update prefreence parameters to", pref_param)

        return pref_param


# projection to simplex
def simplex_proj(x):
    y = -np.sort(-x)
    sum = 0
    ind = []
    for j in range(len(x)):
        sum = sum + y[j]
        if y[j] + (1 - sum) / (j + 1) > 0:
            ind.append(j)
        else:
            ind.append(0)
    rho = np.argmax(ind)
    delta = (1 - (y[:rho+1]).sum())/(rho+1)
    return np.clip(x + delta, 0, 1)
