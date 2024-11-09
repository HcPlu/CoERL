# -*- coding:utf-8 _*-
"""
=================================================
@Project -> File ：CCERL -> mthread.py
@Author ：HcPlu
@Version: 1.0
@Date: 2023/8/29 14:37
@@Description: 
==================================================
"""


import numpy as np
from tianshou.data import Batch
from joblib import delayed



def eval_policy(policy, envs, env_id=0, n_steps=5000):
    total_reward = 0
    total_cost = 0
    total_steps = 0
    env = envs[env_id]
    env_id = env_id
    obs = env.reset()
    trajectory = []
    for i in range(n_steps):
        total_steps += 1
        # valid_action = env.valid_actions(obs)
        # if env_name in CONTINUOUS_ENVS:
        #     action = policy.predict(np.array(env.state_phi(obs)).reshape(1, -1),valida_action, scale="tanh")
        # else:
        input = Batch(obs=np.array([obs]), info={})
        action = policy(input)
        action = action.act
        action = action.cpu().data.numpy()
        # action = policy.predict(np.array(env.state_phi(obs)).reshape(1, -1), valid_action, scale="softmax")
        new_obs, reward, done, info = env.step(action)
        total_reward = total_reward + reward

        trajectory.append(Batch(obs=np.array(obs), act=action[0], rew=reward
              , done=done, obs_next=np.array(new_obs), info=info))
        obs = new_obs

        if done:
            break

    return total_reward, total_cost, total_steps, env_id,trajectory

def eval_policy_env(policy, env, n_steps=5000):
    total_reward = 0
    total_steps = 0
    obs = env.reset()
    trajectory = []
    for i in range(n_steps):
        total_steps += 1
        input = Batch(obs=np.array([obs]), info={})
        action = policy(input)
        action = action.act
        action = action.cpu().data.numpy()
        new_obs, reward, done, info = env.step(action)
        total_reward = total_reward + reward
        trajectory.append(Batch(obs=np.array(obs), act=action[0], rew=reward
              , done=done, obs_next=np.array(new_obs), info=info))
        obs = new_obs

        if done:
            break

    return total_reward, total_steps,trajectory

def eval_policy_env_id(policy, env, n_steps=5000,id=0):
    total_reward = 0
    total_steps = 0
    obs = env.reset()
    trajectory = []
    for i in range(n_steps):
        total_steps += 1
        input = Batch(obs=np.array([obs]), info={})
        action = policy(input)
        action = action.act
        action = action.cpu().data.numpy()
        new_obs, reward, done, info = env.step(action)
        total_reward = total_reward + reward
        trajectory.append(Batch(obs=np.array(obs), act=action[0], rew=reward
              , done=done, obs_next=np.array(new_obs), info=info))
        obs = new_obs

        if done:
            break

    return total_reward, total_steps,trajectory,id


def add_step(buffer,trajectory,device):
    for batch in trajectory:
        buffer.add(batch)

def handle_mix_result(result):
    env_ids = result[:,3]
    keys = np.unique(env_ids)
    split_result = {}
    split_result_idx = {}
    for key in keys:
        split_result[int(key)] = []
        split_result_idx[int(key)] = []
    for idx,single_result in enumerate(result):
        key = int(single_result[3])
        split_result[key].append(single_result)
        split_result_idx[key].append(idx)
    return split_result,split_result_idx
# for parallel

def add_train_log(logger,train_result,step,epoch):
    # split_result,_ = handle_mix_result(train_result)
    # for k, v in train_result.items():
    #     v = np.array(v)
    #     rewards = v[:, 0]
    #     steps = v[:, 2]
    rewards = train_result,
    logger.write("train", step, {
        "train/rewards" : np.mean(rewards),
        "train/rewards_std" : np.std(rewards),
    })
def add_test_log(logger,test_result,step,epoch):
    # split_result,_ = handle_mix_result(test_result)
    rewards = test_result
    logger.write("test",step,{
        "test/rewards": np.mean(rewards),
        "test/rewards_std": np.std(rewards),
    })

eval_policy_delayed = delayed(eval_policy)
