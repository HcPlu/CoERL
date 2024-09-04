# -*- coding:utf-8 _*-
"""
=================================================
@Project -> File ：CCERL -> CCERL.py
@Author ：HcPlu
@Version: 1.0
@Date: 2023/9/8 16:09
@@Description: 
==================================================
"""

import time
from policy.cces import cces_policy
from mthread import eval_policy_delayed, eval_policy, add_step, add_test_log, add_train_log, eval_policy_env
from collections import defaultdict
from joblib import Parallel
import argparse
import os, datetime
from policy.util import hard_copy
import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import tianshou as ts
import tqdm
from tianshou.policy import SACPolicy
# from policy.SACPolicy import SACPolicy
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from policy.simpleSAC import actor_constructor
from policy.simpleSAC import simpleSACPolicy

torch.set_num_threads(1)
gym.logger.set_level(40)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default="HalfCheetah-v2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--buffer-size", type=int, default=1000000)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[256, 256])
    parser.add_argument("--actor-lr", type=float, default=1e-3)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--auto-alpha", default=False, action="store_true")
    parser.add_argument("--alpha-lr", type=float, default=3e-4)
    parser.add_argument("--start-timesteps", type=int, default=10000)
    parser.add_argument("--epoch", type=int, default=500)
    parser.add_argument("--step-per-epoch", type=int, default=2000)
    parser.add_argument("--step-per-collect", type=int, default=1)
    parser.add_argument("--update-per-step", type=int, default=1)
    parser.add_argument("--n-step", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument('--training-num', type=int, default=24)
    parser.add_argument('--test-num', type=int, default=10)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.0)
    parser.add_argument('--rew-norm', action="store_true", default=False)
    parser.add_argument('--es_lr', type=float, default=0.001)
    parser.add_argument('--es_std', type=float, default=0.01)
    parser.add_argument('--es_pop', type=int, default=6)
    parser.add_argument('--es_rl_inject', type=int, default=1)
    parser.add_argument('--utility',type=int, default=True)

    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--save_step', type=int, default=5)
    parser.add_argument('--eval_step', type=int, default=1)
    parser.add_argument('--alg', type=str, default='sac')
    args = parser.parse_known_args()[0]
    return args


def test_discrete_sac(args=get_args()):
    args.device = "cuda:%d" % args.gpu_id
    env = gym.make(args.task)
    env.seed(args.seed)
    print(env.observation_space)

    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    print("Action range:", np.min(env.action_space.low), np.max(env.action_space.high))
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # model
    net_a = Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device)

    actor = ActorProb(
        net_a,
        args.action_shape,
        device=args.device,
        unbounded=True,
        conditioned_sigma=True,
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    net_c1 = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device,
    )
    net_c2 = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device,
    )
    critic1 = Critic(net_c1, device=args.device).to(args.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2 = Critic(net_c2, device=args.device).to(args.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    if args.auto_alpha:
        target_entropy = -np.prod(env.action_space.shape)
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        args.alpha = (target_entropy, log_alpha, alpha_optim)

    policy = SACPolicy(
        actor,
        actor_optim,
        critic1,
        critic1_optim,
        critic2,
        critic2_optim,
        tau=args.tau,
        gamma=args.gamma,
        alpha=args.alpha,
        estimation_step=args.n_step,
        action_space=env.action_space,
    )

    replay_buffer = ts.data.ReplayBuffer(args.buffer_size)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_path = os.path.join(args.logdir, args.task, args.alg, str(args.seed))
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer)
    ac = actor_constructor(args.state_shape, args.action_shape, args.hidden_sizes)
    simple_actor = ac.build_actor("cpu")
    cces = cces_policy(simple_actor,learning_rate=args.es_lr, noise_std=args.es_std,utility=args.utility)
    best_reward = -1e8
    collected_steps = 0

    pop_num = args.es_pop

    st = time.time()
    for ep in range(args.epoch):
        t = 0
        group_num = np.random.choice([2, 3, 4])
        with tqdm.tqdm(total=args.step_per_epoch) as _tdqm:
            _tdqm.set_description("epoch:{}/{}".format(ep + 1, args.epoch))
            if ep % args.save_step == 0:
                ckpt_path = os.path.join(log_path, f"checkpoint_{ep}.pth")
                torch.save(policy.state_dict(), ckpt_path)

            if ep % args.eval_step == 0:
                hard_copy(cces.actor, policy.cpu())
                jobs = [cces.pool.apply_async(eval_policy_env, (cces.actor, env,)) for pop_id in range(10)]
                test_rewards = []
                for idx, j in enumerate(jobs):
                    test_rewards.append(j.get()[0])

                test_rewards = np.array(test_rewards)
                add_test_log(logger, test_rewards, collected_steps, ep)

                print(ep, {"test_rew": np.mean(test_rewards), })
                if best_reward < np.mean(test_rewards):
                    best_reward = np.mean(test_rewards)
                    torch.save(policy.state_dict(), os.path.join(log_path, 'best.pth'))

            # training
            while t < args.step_per_epoch:
                # CCES
                es_actor, rewards, timesteps_count, es_trajs = cces.train(actor=policy,
                                                                          group_num=group_num,
                                                                          pop_num=pop_num, env=env)
                print( "mean:", np.mean(rewards), "std:",np.std(rewards), "max:", np.max(rewards),timesteps_count )
                ep_steps = timesteps_count
                collected_steps += ep_steps
                t += ep_steps
                # replace the rl actor with updated es actor
                hard_copy(policy, es_actor)
                rewards = np.array(rewards)
                trajs = []
                for item in es_trajs:
                    trajs += item
                add_step(replay_buffer, trajs,args.device)
                logger.write("train", collected_steps,
                             {"es_group_num":group_num,})
                logger.write("train", collected_steps,
                             {"train/es_rewards":np.mean(rewards),"train/es_rewards_std":np.std(rewards)})
                #periodically inject RL
                if ep % args.es_rl_inject == 0:
                    policy.to(args.device)
                    if replay_buffer.__len__() > args.start_timesteps:
                        policy.train()
                        losses_stat = []

                        for _ in range(int(ep_steps * args.update_per_step)):
                            losses = policy.update(args.batch_size, replay_buffer)
                            losses_stat.append(losses)

                        stats = defaultdict(list)
                        for item in losses_stat:
                            for k, v in item.items():
                                stats[k].append(v)
                        for k, v in stats.items():
                            stats[k] = round(np.mean(v), 4)

                        hard_copy(cces.actor, policy.cpu())
                        jobs = [cces.pool.apply_async(eval_policy_env, (cces.actor, env,)) for pop_id in range(10)]
                        rl_rewards = []
                        for idx, j in enumerate(jobs):
                            rl_rewards.append(j.get()[0])

                        rl_rewards = np.array(rl_rewards)
                        logger.write("train", collected_steps,
                                     {"train/rl_rewards": np.mean(rl_rewards), "train/rl_rewards_std": np.std(rl_rewards)})



                        logger.write("loss", collected_steps, stats)
                        res_data = {"rew": np.mean(rewards), "steps": collected_steps,"FPS":collected_steps / (time.time() - st)}





                        stats.update(res_data)
                        _tdqm.set_postfix(stats)
            _tdqm.update(ep_steps)

        if collected_steps>args.epoch*args.step_per_epoch:
            if ep % args.save_step == 0:
                ckpt_path = os.path.join(log_path, f"checkpoint_{ep}.pth")
                torch.save(policy.state_dict(), ckpt_path)

            if ep % args.eval_step == 0:
                hard_copy(cces.actor, policy.cpu())
                jobs = [cces.pool.apply_async(eval_policy_env, (cces.actor, env,)) for pop_id in range(10)]
                test_rewards = []
                for idx, j in enumerate(jobs):
                    test_rewards.append(j.get()[0])

                test_rewards = np.array(test_rewards)
                add_test_log(logger, test_rewards, collected_steps, ep)

                print(ep, {"test_rew": np.mean(test_rewards), })
                if best_reward < np.mean(test_rewards):
                    best_reward = np.mean(test_rewards)
                    torch.save(policy.state_dict(), os.path.join(log_path, 'best.pth'))
            break

    print({"collected_step": collected_steps, "best_reward": best_reward})
    torch.save(policy.state_dict(), os.path.join(log_path, 'final.pth'))


if __name__ == '__main__':
    torch.multiprocessing.set_start_method("spawn")
    test_discrete_sac()