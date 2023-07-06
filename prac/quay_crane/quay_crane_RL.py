# 使用深度神经网络
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


np.random.seed(1)
torch.manual_seed(1)


# define the network architecture
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


# 隐藏层的层数
class DeepQNetwork():
    def __init__(self, n_actions, n_features, n_hidden=26, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9,
                 replace_target_iter=200, memory_size=500, batch_size=32, e_greedy_increment=None,
                 ):
        self.n_actions = n_actions    # 输出层多少个动作
        self.n_features = n_features  # 接收多少个observation
        self.n_hidden = n_hidden
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy     # 以0.9概率选择
        self.replace_target_iter = replace_target_iter  # 隔了多少步将target更新为最新的参数
        self.memory_size = memory_size  # 记忆库中的容量（此处设置为多少最优？？）
        self.batch_size = batch_size    # 神经网络梯度学习中使用
        self.epsilon_increment = e_greedy_increment  # 不断缩小随机范围
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        # 上个语句理解如果是None，执行0，如果不是的话，一直为0.9概率选择
        # e_greedy_increment 是不断的缩小随机范围

        # total learning step  整体学习步数
        self.learn_step_counter = 0   # 最开始置为0

        # initialize zero memory [s, a, r, s_]  乘2是两个输入状态，加2是动作、奖励值
        # 此处也可以使用pandas
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        self.loss_func = nn.MSELoss()
        self.cost_his = []  # 记录每一步的误差
        self._build_net()

    def _build_net(self):
        # 建立两个相同的神经结构
        self.q_eval = Net(self.n_features, self.n_hidden, self.n_actions)     # 评估网络
        self.q_target = Net(self.n_features, self.n_hidden, self.n_actions)   # 目标网络
        # 只单独对估计网络进行学习
        self.optimizer = torch.optim.RMSprop(self.q_eval.parameters(), lr=self.lr)  # 神经网络优化算法

    # 优化器逐渐降低与loss差距

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):   # 如果类中没有该属性，则计数为0
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size  # 如果满了从头开始存
        self.memory[index, :] = transition  # 将记忆库数据堆叠
        self.memory_counter += 1

    def choose_action(self, observation):
        observation = torch.Tensor(observation[np.newaxis, :])  # 为了处理，将维度上升（升维度的意义）
        if np.random.uniform() < self.epsilon:            # 此时应该随着训练增多降低探索概率
            actions_value = self.q_eval(observation)      # 放到估计网络中进行分析（使用的是tensor）
            action = np.argmax(actions_value.data.numpy())  # 将数据转化
        else:
            action = np.random.randint(0, self.n_actions)  # 如何降低此处探索概率？？
        return action

    def learn(self):
        # check to replace target parameters  更新目标网络参数
        # 如果进行更换就更换，如果不进行更换就跳过
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.q_target.load_state_dict(self.q_eval.state_dict())  # 将参数赋值
            print("\ntarget params replaced\n")   # 目标网络参数被替换
          # print(self.q_eval.state_dict())    # 权重参数以及偏置参数

        # sample batch memory from all memory
        # 开始调用记忆库中记忆，如果记忆库记忆不够，抽取已经存下的记忆
        # 此处是步数至少两百步（存储了两百条），不满的才可以顺利运行
        # 选取一批记忆，此处是32个
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)  # 随机选择
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        # q_next is used for getting which action would be choosed by target network in state s_(t+1)
        # 此处与transition中怎么存储记忆有关系
        q_next = self.q_target(torch.Tensor(batch_memory[:, -self.n_features:]))  # 后状态，t+1的状态
        q_eval = self.q_eval(torch.Tensor(batch_memory[:, :self.n_features]))     # 前状态，t状态
        # used for calculating y, we need to copy for q_eval because this operation could keep the Q_value that has not been selected unchanged,
        # so when we do q_target - q_eval, these Q_value become zero and wouldn't affect the calculation of the loss
        q_target = torch.Tensor(q_eval.data.numpy().copy())

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = torch.Tensor(batch_memory[:, self.n_features + 1])
        q_target[batch_index, eval_act_index] = reward + self.gamma * torch.max(q_next, 1)[0]

        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # increase epsilon   此处没有增加
        self.cost_his.append(loss)
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1


#
    def plot_cost(self):
        for i in range(len(self.cost_his)):
            self.cost_his[i] = self.cost_his[i].detach().numpy()  # 去除列表中每项tensor的梯度
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.savefig('C:/Users/89317/Desktop/实验图片/loss_memory_3000_1（多个集装箱1_5000）.jpg', bbox_inches='tight')
        plt.show()

    # 保存训练好的神经网络参数
    #def save_parameter(self):
    # 既保存网络结构也保存参数
    #torch.save(q_eval, 'net.pkl')
    # 只保存神经网络训练模型参数
        #torch.save(.state_dict(), 'net_params.pkl')

