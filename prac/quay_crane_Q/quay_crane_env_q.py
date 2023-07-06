# 本环境时间步假设情况较为理想

# 学习环境是如何搭建的
# 操作时间
# quay_time_work = [6]   # 单位是分钟，平均作业时间
# quay_time_move = [2]   # 移动一个贝位时间
# 移动和工作的时间未考虑（如何考虑），应该从设计奖励角度出发
#

import time
import numpy as np  # 导入numpy


class Envir(object):
    def __init__(self):
        self.action_space = ['ll', 'lr', 'lw', 'rl', 'rr', 'rw', 'wl', 'wr', 'ww']  # 九种工作状态
        self.n_actions = len(self.action_space)
        self.initialization()
        self.n_features = len(self.set)  # 输入状态的维度数量

#  参数初始化
    def initialization(self):
        # 集装箱任务
        # 0无箱 1 有箱未被操作
        self.container_position = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        self.quay_crane_position1 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.quay_crane_position2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        # self.count = np.array(self.container_position).sum()
        # 所有状态形式合并
        # 数据变成一维向量
        self.c_p_len = len(self.container_position)
        self.q_c_p1_len = len(self.container_position + self.quay_crane_position1)
        self.q_c_p2_len = len(self.container_position + self.quay_crane_position1 + self.quay_crane_position2)

        self.set = self.container_position + self.quay_crane_position1 + self.quay_crane_position2



    def reset(self):
        time.sleep(0.01)  # 卡个时间
        self.initialization()
        self.set_numpy = np.array(self.set)  # 将数据变成数组形式
        return self.set_numpy


    def step(self, action):
        # 一维情况
        c_p = self.set[0:self.c_p_len]
        q_c_p1 = self.set[self.c_p_len:self.q_c_p1_len]
        q_c_p2 = self.set[self.q_c_p1_len:self.q_c_p2_len]


        if action == 0 :  # 左左
            if q_c_p1[0] != 1:       # 走通会改，没有走通就不变
               max_vaule1 = max(q_c_p1)  # 选最大值
               max_idx1 = q_c_p1.index(max_vaule1)  # 找最大值索引
               q_c_p1[max_idx1] = 0      # 置为0
               q_c_p1[max_idx1-1] = 1    # 向左移动

               if q_c_p2[0] != 1:    # 满足前述条件执行此指令,不满足就不变
                   max_vaule2 = max(q_c_p2)
                   max_idx2 = q_c_p2.index(max_vaule2)
                   q_c_p2[max_idx2] = 0
                   q_c_p2[max_idx2 - 1] = 1

        elif action == 1:  # 左右
            if q_c_p1[0] != 1:
               max_vaule1 = max(q_c_p1)
               max_idx1 = q_c_p1.index(max_vaule1)
               q_c_p1[max_idx1] = 0
               q_c_p1[max_idx1-1] = 1

               if q_c_p2[-1] != 1:
                   max_vaule2 = max(q_c_p2)
                   max_idx2 = q_c_p2.index(max_vaule2)
                   q_c_p2[max_idx2] = 0
                   q_c_p2[max_idx2 + 1] = 1

        elif action == 2:  # 左工
            if q_c_p1[0] != 1:
               max_vaule1 = max(q_c_p1)
               max_idx1 = q_c_p1.index(max_vaule1)
               q_c_p1[max_idx1] = 0
               q_c_p1[max_idx1-1] = 1

               max_vaule2 = max(q_c_p2)
               max_idx2 = q_c_p2.index(max_vaule2)
               if c_p[max_idx2] == 1:  # 判断此处是否有箱
                   c_p[max_idx2] = 0   # 有箱就变，无箱就不变

        elif action == 3:  # 右左
            # 不需要判断  将交叉变成惩罚（看是否可行）
            max_vaule1 = max(q_c_p1)
            max_idx1 = q_c_p1.index(max_vaule1)
            q_c_p1[max_idx1] = 0
            q_c_p1[max_idx1 + 1] = 1

            max_vaule2 = max(q_c_p2)
            max_idx2 = q_c_p2.index(max_vaule2)
            q_c_p2[max_idx2] = 0
            q_c_p2[max_idx2 - 1] = 1

        elif action == 4:  # 右右
            max_vaule1 = max(q_c_p1)
            max_idx1 = q_c_p1.index(max_vaule1)
            q_c_p1[max_idx1] = 0
            q_c_p1[max_idx1 + 1] = 1

            if q_c_p2[-1] != 1:
              max_vaule2 = max(q_c_p2)
              max_idx2 = q_c_p2.index(max_vaule2)
              q_c_p2[max_idx2] = 0
              q_c_p2[max_idx2 - 1] = 1

        elif action == 5:  # 右工
            max_vaule1 = max(q_c_p1)
            max_idx1 = q_c_p1.index(max_vaule1)
            q_c_p1[max_idx1] = 0
            q_c_p1[max_idx1 + 1] = 1

            max_vaule2 = max(q_c_p2)
            max_idx2 = q_c_p2.index(max_vaule2)
            if c_p[max_idx2] == 1:
                c_p[max_idx2] = 0

        elif action == 6:  # 工左
            max_vaule1 = max(q_c_p1)
            max_idx1 = q_c_p1.index(max_vaule1)
            if c_p[max_idx1] == 1:
                c_p[max_idx1] = 0

            max_vaule2 = max(q_c_p2)
            max_idx2 = q_c_p2.index(max_vaule2)
            q_c_p2[max_idx2] = 0
            q_c_p2[max_idx2 - 1] = 1

        elif action == 7:  # 工右
            max_vaule1 = max(q_c_p1)
            max_idx1 = q_c_p1.index(max_vaule1)
            if c_p[max_idx1] == 1:
                c_p[max_idx1] = 0

            if q_c_p2[-1] != 1:
                max_vaule2 = max(q_c_p2)
                max_idx2 = q_c_p2.index(max_vaule2)
                q_c_p2[max_idx2] = 0
                q_c_p2[max_idx2 - 1] = 1

        elif action == 8:  # 工工
            max_vaule1 = max(q_c_p1)
            max_idx1 = q_c_p1.index(max_vaule1)
            if c_p[max_idx1] == 1:
                c_p[max_idx1] = 0

            max_vaule2 = max(q_c_p2)
            max_idx2 = q_c_p2.index(max_vaule2)
            if c_p[max_idx2] == 1:
                c_p[max_idx2] = 0

        s_ = np.array(c_p + q_c_p1 + q_c_p2)

        if s_[0:self.c_p_len].sum() == 0:  # 所有集装箱总数
            reward = 1
            done = True
            s_ = 'terminal'
        # 最少相隔一个贝位
        elif abs(list(s_[self.c_p_len:self.q_c_p1_len]).index(max(list(s_[self.c_p_len:self.q_c_p1_len])))
            -list(s_[self.q_c_p1_len:self.q_c_p2_len]).index(max(list(s_[self.q_c_p1_len:self.q_c_p2_len])))) <= 2:
            reward = -1
            done = True
            s_ = 'terminal'
            # pass
        else:
            reward = 0
            done = False  # 训练使用
            # done = True   # 测试代码使用

        return s_, reward, done

    # 刷新环境
    def render(self):
        time.sleep(0.01)

def update():
    for t in range(1):
        s = env.reset()  # 循环开始前需要初始化 所有内容初始化
        while True:
            env.render()
            a = 6
            s, r, done =env.step(a)
            print(s)
            if done:
                break


if __name__ == '__main__':
    env = Envir()
    update()
    #print(update.s)
    print('程序结束')




# 参数初始化中
 # 数据变成二维向量
        # self.set = [self.container_position, self.quay_crane_position1, self.quay_crane_position2]

# step中
# 二维情况
        # c_p = self.set[0]
        # q_c_p1 = self.set[1]
        # q_c_p2 = self.set[2]
        # 从左往右看

# s_ = [c_p, q_c_p1, q_c_p2]



