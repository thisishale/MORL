from __future__ import absolute_import, division, print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


class EnvelopeLinearCQN(torch.nn.Module):
    '''
        Linear Controllable Q-Network, Envelope Version
    '''

    def __init__(self, state_size, action_size, reward_size):
        super(EnvelopeLinearCQN, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.reward_size = reward_size

        # S x A -> (W -> R^n). =>. S x W -> (A -> R^n)
        self.affine1 = nn.Linear(state_size + reward_size,
                                 (state_size + reward_size) * 16)
        self.affine2 = nn.Linear((state_size + reward_size) * 16,
                                 (state_size + reward_size) * 32)
        self.affine3 = nn.Linear((state_size + reward_size) * 32,
                                 (state_size + reward_size) * 64)
        self.affine4 = nn.Linear((state_size + reward_size) * 64,
                                 (state_size + reward_size) * 32)
        self.affine5 = nn.Linear((state_size + reward_size) * 32,
                                 action_size * reward_size)
    def H(self, Q, w, s_num, w_num):
        # Q [2,6]
        # w [1,6]
        # s_num is 0
        # w_num is 0
        mask = torch.cat(
            [torch.arange(i, s_num * w_num + i, s_num)
             for i in range(s_num)]).type(LongTensor)
        
        reQ = Q.view(-1, self.action_size * self.reward_size
                     )[mask].view(-1, self.reward_size)
        # reQ and Q are the same here.
        # reQ is of size [2, 6]
        # extend Q batch and preference batch
        reQ_ext = reQ.repeat(w_num, 1)
        # reQ_ext is of size [2, 6]
        w_ext = w.unsqueeze(2).repeat(1, self.action_size * w_num, 1)
        # w_ext is of size [1, 12, 1]
        w_ext = w_ext.view(-1, self.reward_size)
        # w_ext is of size [2, 6]
        # w_ext is like w but repeated in 0 axis.
        # produce the inner products
        # reQ_ext.unsqueeze(1) of size [2, 1, 6]
        # w_ext.unsqueeze(2) of size [2, 6, 1]
        prod = torch.bmm(reQ_ext.unsqueeze(1), w_ext.unsqueeze(2)).squeeze()
        # prod is of shape [2]
        # mask for take max over actions and weights
        prod = prod.view(-1, self.action_size * w_num)
        # prod [1, 2]
        inds = prod.max(1)[1]
        mask = ByteTensor(prod.size()).zero_()
        mask.scatter_(1, inds.data.unsqueeze(1), 1)
        mask = mask.view(-1, 1).repeat(1, self.reward_size)
        # inds is either 0 or 1.
        # mask is either 
        # [[1, 1, 1, 1, 1, 1],
        # [0, 0, 0, 0, 0, 0]]
        # or
        # [[0, 0, 0, 0, 0, 0],
        # [1, 1, 1, 1, 1, 1]] 
        # depending on inds.
        # get the HQ
        # HQ is one the rows of reQ_ext.
        # print(mask)
        HQ = reQ_ext.masked_select(Variable(mask)).view(-1, self.reward_size)
        # so here we get the action row of q that corresponds to max w.q
        return HQ

    def H_(self, Q, w, s_num, w_num):
        reQ = Q.view(-1, self.reward_size)

        # extend preference batch
        w_ext = w.unsqueeze(2).repeat(1, self.action_size, 1).view(-1, 2)

        # produce hte inner products
        prod = torch.bmm(reQ.unsqueeze(1), w_ext.unsqueeze(2)).squeeze()

        # mask for take max over actions
        prod = prod.view(-1, self.action_size)
        inds = prod.max(1)[1]
        mask = ByteTensor(prod.size()).zero_()
        mask.scatter_(1, inds.data.unsqueeze(1), 1)
        mask = mask.view(-1, 1).repeat(1, self.reward_size)

        # get the HQ
        HQ = reQ.masked_select(Variable(mask)).view(-1, self.reward_size)

        return HQ

    def forward(self, state, preference, w_num=1):
        # print(state.shape) [1,2]
        # if hq is extracted from this method then:
            # pref >>> [1] its been unsqueezed
            # state >>> [1] its been unsqueezed
        # else:
            # pref >>> [1, 6]
            # state >>> [1, 2]
        s_num = int(preference.size(0) / w_num)
        x = torch.cat((state, preference), dim=1)
        # print(x.shape) [1,8]
        x = x.view(x.size(0), -1)
        # print(x.shape) [1,8]
        x = F.relu(self.affine1(x))
        # print(x.shape) [1,128]
        x = F.relu(self.affine2(x))
        # print(x.shape) [1,256]
        x = F.relu(self.affine3(x))
        # print(x.shape) [1,512]
        x = F.relu(self.affine4(x))
        # print(x.shape) [1,256]
        q = self.affine5(x)
        # print(q.shape)
        # print(q.shape) [1,12]
        q = q.view(q.size(0), self.action_size, self.reward_size)
        # [1,2,6]
        # s_num = 6
        # w_num = 1
        hq = self.H(q.detach().view(-1, self.reward_size), preference, s_num, w_num)

        return hq, q
