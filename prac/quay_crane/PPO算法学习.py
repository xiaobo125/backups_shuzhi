"""PPO（Proximal  Policy  Optimization）算法的主要思想是通过比较两个策略的差异来更新策略。具体来说，
它使用一个截断的目标函数来限制每次更新的大小，以避免步长过大造成的不完全收敛和过拟合问题。

简单原理理解：

PPO  算法采用了最大熵强化学习算法和  TRPO  算法的一些方法，较好地解决了在强化学习实践中出现的一些问题
，如：样本利用不充分的问题、优化过程不稳定易发散等问题，同时表现效果更加优异。PPO  优化算法的主要目的是
优化强化学习中的策略函数，使其最大化目标策略，同时保证新策略与旧策略之间的  KL  散度小于一个门限值，
以此来限制策略函数的优化速度，保证训练收敛性和计算效率。具体而言，其过程包括三个主要步骤：数据采集、多步回归
和策略更新，其中数据采集可以利用已有的策略或者随机策略进行采样，多步回归则是使用代理目标函数优化估计的策略价值函数，
策略更新则是根据优化的目标函数来更新策略，同时保证更新时新策略尽量遵循旧策略的一些特定约束条件，
目的是为了避免卡在一个瓶颈上，从而选择权重参数最优的更新方案。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque
import random
import numpy as np



class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        super(ActorCritic, self).__init__()
        self.seed = torch.manual_seed(seed)  # 设置随机数种子
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3_actor = nn.Linear(fc2_units, action_size)
        self.fc3_critic = nn.Linear(fc2_units, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actor = self.fc3_actor(x)
        critic = self.fc3_critic(x)

        return actor, critic


class Agent():
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64, lr=1e-4,
                 update_every=4, gamma=0.99,   # 每隔几个时间步更新一次神经网络参数  # 折扣率
                 tau=0.95, eps_clip=0.2):   # 用于软更新目标是神经网络参数的参数
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.eps_clip = eps_clip

        self.actor_critic = ActorCritic(state_size, action_size, seed, fc1_units, fc2_units)
        self.actor_critic_target = ActorCritic(state_size, action_size, seed, fc1_units, fc2_units)
        self.optimizer = nn.optimizer.Adam(self.actor_critic.parameters(), lr=lr)

        self.memory = deque()
        self.update_every = update_every
        self.step_count = 0

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        actor, _ = self.actor_critic(state)
        dist = Categorical(F.softmax(actor, dim=1))
        action = dist.sample()

        return action.item(), dist.log_prob(action)

    def step(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        self.step_count += 1
        if self.step_count % self.update_every == 0:
            self.learn()

    def learn(self):
        states, actions, rewards, next_states, dones = zip(*self.memory)
        states = torch.from_numpy(np.vstack(states)).float()
        actions = torch.from_numpy(np.vstack(actions)).long()
        rewards = torch.from_numpy(np.vstack(rewards)).float()
        next_states = torch.from_numpy(np.vstack(next_states)).float()
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float()

        _, current_value = self.actor_critic(states)
        _, next_value = self.actor_critic(next_states)
        target = rewards + (self.gamma * next_value * (1 - dones))
        adv = target - current_value

        old_actor, _ = self.actor_critic(states)
        dist_old = Categorical(F.softmax(old_actor, dim=1))
        old_log_probs = dist_old.log_prob(actions)

        for _ in range(self.optimizer_steps):
            new_actor, new_value = self.actor_critic(states)
            new_actor_target, new_value_target = self.actor_critic_target(next_states)

            new_dist = Categorical(F.softmax(new_actor, dim=1))
            new_log_probs = new_dist.log_prob(actions)
            ratios = torch.exp(new_log_probs - old_log_probs)

            surr1 = ratios * adv
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * adv

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(new_value, target.detach())

            self.optimizer.zero_grad()
            (actor_loss + critic_loss).backward()
            self.optimizer.step()

            with torch.no_grad():
                for current, target in zip(self.actor_critic.parameters(), self.actor_critic_target.parameters()):
                    target.data.copy_(self.tau * current.data + (1.0 - self.tau) * target.data)

        self.memory = deque()
        self.step_count = 0
