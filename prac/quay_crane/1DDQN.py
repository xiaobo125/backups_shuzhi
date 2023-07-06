import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

np.random.seed(1)
torch.manual_seed(1)


# 标准神经网络结构
class Net(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.el = nn.Linear(n_feature, n_hidden)
        self.q = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = self.el(x)
        x = F.relu(x)
        x = self.q(x)
        return x


class DoubleDQN():
    def __init__(self, n_actions, n_features, n_hidden=20, learning_rate=0.005, reward_decay=0.9, e_greedy=0.9,
                 replace_target_iter=200, memory_size=3000, batch_size=32, e_greedy_increment=None, double_q=True):
        self.n_actions = n_actions
        self.n_hidden = n_hidden
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.double_q = double_q

        self.learn_step_counter = 0
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))
        self._build_net()
        self.cost_his = []

    def _build_net(self):
        self.q_eval = Net(self.n_features, self.n_hidden, self.n_actions)
        self.q_target = Net(self.n_features, self.n_hidden, self.n_actions)
        self.optimizer = torch.optim.RMSprop(self.q_eval.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        observation = torch.Tensor(observation[np.newaxis, :])
        actions_value = self.q_eval(observation)
        action = torch.max(actions_value, dim=1)[1]  # record action value it get
        if not hasattr(self, 'q'):
            self.q = []
            self.running_q = 0
        self.running_q = self.running_q * 0.99 + 0.01 * torch.max(actions_value, dim=1)[0]
        self.q.append(self.running_q)

        if np.random.uniform() > self.epsilon:  # randomly choose action
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.q_target.load_state_dict(self.q_eval.state_dict())
            print("\ntarget params replaced\n")

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)

        batch_memory = self.memory[sample_index, :]

        # q_eval4next is the output of the q_eval network when input s_(t+1)
        # q_next is the output of the q_target network when input s_(s+1)
        # we use q_eval4next to get which action was choosed by eval network in s_(t+1)
        # then we get the Q_value corresponding to that action output by target network
        q_next, q_eval4next = self.q_target(torch.Tensor(batch_memory[:, -self.n_features:])), self.q_eval(
            torch.Tensor(batch_memory[:, -self.n_features:]))
        q_eval = self.q_eval(torch.Tensor(batch_memory[:, :self.n_features]))

        # used for calculating y, we need to copy for q_eval because this operation could keep the Q_value that has not been selected unchanged,
        # so when we do q_target - q_eval, these Q_value become zero and wouldn't affect the calculation of the loss
        q_target = torch.Tensor(q_eval.data.numpy().copy())

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = torch.Tensor(batch_memory[:, self.n_features + 1])

        if self.double_q:
            max_act4next = torch.max(q_eval4next, dim=1)[1]
            selected_q_next = q_next[batch_index, max_act4next]
        else:
            selected_q_next = torch.max(q_next, dim=1)[0]

        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next

        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.cost_his.append(loss)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1


from quay_crane_env import Envir
import matplotlib.pyplot as plt
import time

def run_Envir():
    step = 0
    count_x = []
    count_figure = []
    episode_sum = 2000          # 迭代次数
    for episode in range(episode_sum):  # 迭代三百次
        episode_reward_sum = 0
        print("episode: {}".format(episode))
        observation = env.reset()  # 初始化
        while True:
            # print("step: {}".format(step))
            env.render()
            action = RL.choose_action(observation)
            print("选取动作: {}".format(action))
            observation_, reward, done = env.step(action)
            #print(reward)
            RL.store_transition(observation, action, reward, observation_)
            episode_reward_sum += reward
            if (step > 200) and (step % 5 == 0):
                RL.learn()
            observation = observation_
            if done:
                print('episode%s---reward_sum: %s' % (episode, episode_reward_sum))   # 输出对应奖励
                count_figure.append(episode_reward_sum)
                break
            step += 1
    print('game over')
    for i in range(episode_sum):
        count_x.append(i)
    plt.plot(count_x, count_figure)
    plt.savefig('C:/Users/89317/Desktop/实验图片/reward_memory_3000_1.jpg', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    time_start = time.time()  # 记录开始时间
    env = Envir()
    RL = DoubleDQN(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=3000  # 原本两千
                      )
    run_Envir()
    RL.plot_cost()
    time_end = time.time()  # 记录结束时间
    time_sum_second = (time_end - time_start)  # 单位为秒
    time_sum_minute = (time_end - time_start) / 60
    time_sum_hour = (time_end - time_start) / 3600  # 计算的时间差为程序的执行时间，单位为小时/h
    print(time_sum_second)  # 计算花费秒
    print(time_sum_minute)
    print(time_sum_hour)  # 计算花费小时
