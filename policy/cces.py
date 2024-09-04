# -*- coding:utf-8 _*-
"""
=================================================
@Project -> File ：CCERL -> cc.py
@Author ：HcPlu
@Version: 1.0
@Date: 2023/8/29 14:35
@@Description: 
==================================================
"""
import time
from policy.simpleSAC import actor_constructor
import numpy as np
from copy import deepcopy
import math
import torch
from mthread import eval_policy_env
from policy.util import decode_model,encode_model,hard_copy
import torch.multiprocessing as mp
torch.set_num_threads(1)
class SGD():
    def __init__(self,actor,momentum=0.9,weight_decay=0.0001):
        self.actor = actor
        self.params_shapes = [param.shape for param in actor.parameters()]
        self.momentum = momentum
        self.v = [torch.zeros_like(param) for param in actor.parameters()]

    def get_gradient(self,i,gradient):
        self.v[i] = self.momentum*self.v[i] + gradient*(1-self.momentum)
        return self.v[i]

class cces_policy():
    def __init__(self, actor,learning_rate=0.05, noise_std=0.2,noise_decay=1.0, lr_decay=1.0, decay_step=50, utility=True,device='cpu'):
        self.actor = actor
        self.SGD = SGD(actor)
        self._lr = learning_rate
        self._noise_std = noise_std
        self.device = device
        self.noise_decay = noise_decay
        self.lr_decay = lr_decay
        self.decay_step = decay_step
        self.utility = utility
        self._count = 0
        self.params_shapes = [param.shape for param in actor.parameters()]
        self.params_num = np.sum([param.numel() for param in actor.parameters()])
        self.pool = mp.Pool(processes=mp.cpu_count() - 1)
        for param in actor.parameters():
            print(param.shape,param.numel())


    @property
    def noise_std(self):
        step_decay = np.power(self.noise_decay, np.floor((1 + self._count) / self.decay_step))

        return self._noise_std * step_decay

    @property
    def lr(self):
        step_decay = np.power(self.lr_decay, np.floor((1 + self._count) / self.decay_step))

        return self._lr * step_decay
    def _sign(self, x):
        return -1 if x%2==0 else 1


    def generate_pop_sub(self,actor,sub_problem, npop=50):
        pop = []
        flattened_actor = encode_model(actor)
        noise_num = len(sub_problem)
        noise_list = []
        pair_noise_list = []
        #pair sampling
        for pop_i in range(int(npop/2)):
            flatten_noise = torch.zeros(self.params_num)
            noise = torch.randn(noise_num)
            flatten_noise[sub_problem] = noise
            pair_noise_list.append(flatten_noise)
            pair_noise_list.append(-1*flatten_noise)

        for pop_i in range(npop):
            flatten_noise = pair_noise_list[pop_i]
            sub_actor,individual_noise = decode_model(deepcopy(actor),flatten_noise,self.noise_std)
            pop.append(sub_actor)
            noise_list.append(individual_noise)
        return pop,noise_list


    def _utility(self, n):
        _util = np.maximum(0, np.log(n / 2 + 1) - np.log(np.arange(1, n + 1)))
        utility = _util/_util.sum()-1/n
        return utility

    def naive_cc(self,group_num=3):
        random_idx = np.random.permutation(self.params_num)
        slice_list = sorted(np.random.randint(1,self.params_num,group_num-1),reverse=False)
        slice_list.insert(0,0)
        slice_list.append(self.params_num)
        sliced_schedule = [random_idx[slice_list[i]:slice_list[i+1]] for i in range(group_num)]
        # necessary?
        # sliced_schedule = [np.arange(self.params_num)]
        return sliced_schedule


    def train(self,actor,env,pop_num=10,group_num=3):

        hard_copy(self.actor, actor.cpu())
        sliced_schedule = self.naive_cc(group_num=group_num)
        st = time.time()
        rewards = []
        trajs = []
        timesteps_count = 0
        for sub_i,sub_problem in enumerate(sliced_schedule):
            sub_pop,pop_noise = self.generate_pop_sub(self.actor,sub_problem,pop_num)
            # sub_pop = [(item,sub_i) for item in sub_pop]
            jobs = [self.pool.apply_async(eval_policy_env, (sub_pop[pop_id], env,)) for pop_id in range(pop_num)]
            sub_rewards = []
            sub_timesteps = []
            sub_trajs = []
            sub_timesteps_count = 0
            for idx, j in enumerate(jobs):
                sub_rewards.append(j.get()[0])
                sub_timesteps.append(j.get()[1])
                sub_timesteps_count += j.get()[1]
                sub_trajs.append(j.get()[2])
            rewards += sub_rewards
            timesteps_count += sub_timesteps_count
            trajs += sub_trajs
            self._population = sub_pop
            if self.noise_std!=0:
                self.update_actor(self.actor, np.array(sub_rewards), pop_noise)
        return self.actor,rewards,timesteps_count,trajs


    def update_actor(self,actor,rewards,pop_noise,weight_decay=0.):
        if self._population is None:
            raise ValueError("populations is none, generate & eval it first")
        rewards_idx = np.argsort(rewards)
        score = np.zeros(rewards.shape)


        if self.utility:
            score[rewards_idx] = np.arange(rewards.shape[0])[::-1]
            score = score.astype(np.int32)
            utility = self._utility(rewards.shape[0])
            rewards = utility[score]
        else:
            score[rewards_idx] = np.arange(rewards.shape[0])[::-1]
            rewards = (score - score.mean()) / (score.std())

        for i, param in enumerate(actor.parameters()):
            w_updates = torch.from_numpy(np.zeros(param.shape)).to(self.device)
            for j, model in enumerate(self._population):
                w_updates = w_updates + (pop_noise[j][i] * rewards[j])
            w_updates = w_updates.to(self.device)

            param.data = (1-weight_decay)*param.data + ((self.lr / (len(rewards) * self.noise_std)) * w_updates).float()
        self._count = self._count + 1