# 本环境时间步假设情况较为理想
# 操作时间考虑如下
# 需要加机械状态的考虑
# quay_time_work = [6]   # 单位是分钟，平均作业时间
# quay_time_move = [2]   # 移动一个贝位时间
# 移动和工作的时间未考虑（如何考虑），应该从设计奖励角度出发

import time
import numpy as np  # 导入numpy


class Envir(object):
    def __init__(self):
        self.action_space = ['ll', 'lr', 'lw', 'rl', 'rr', 'rw', 'wl', 'wr', 'ww']  # 九种工作状态
        self.n_actions = len(self.action_space)
        self.initialization()            # 任务初始化
        self.n_features = len(self.set)  # 输入状态的维度数量
        self.constraint = 2              # 约束贝位间隔(数字-1是相隔贝位数，此时是2)


    # 直接给定状态
    def initialization(self):
        # 任务状态，0无箱 1 有箱未被操作
        self.container_position = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        # 岸桥状态
        self.quay_crane_position1 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.quay_crane_position2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        # 求每一段的长度
        self.c_p_len = len(self.container_position)
        self.q_c_p1_len = len(self.container_position + self.quay_crane_position1)
        self.q_c_p2_len = len(self.container_position + self.quay_crane_position1 + self.quay_crane_position2)
        # 各自状态合并形式   并变成数组形式
        self.set = np.array(self.container_position + self.quay_crane_position1 + self.quay_crane_position2)
# 实例化类到此所有的结束

# 只有在每个episode开始时使用,将所有任务再进行初始化
    def reset(self):
        time.sleep(0.01)                     # 延迟时间的状态
        self.initialization()                # 重新开始任务
        return self.set                      # 返回状态（数组形式）


# 此处优化一个合法动作模块（不设置负奖励模块）
# 贝位不合法  （边界条件暂且不设置）





    def step(self, action):
        # 有问题，相当于每一次初始化（此处修改）

        c_p = self.set[0:self.c_p_len]
        q_c_p1 = self.set[self.c_p_len:self.q_c_p1_len]
        q_c_p2 = self.set[self.q_c_p1_len:self.q_c_p2_len]
        # 判断此时箱子计步
        set_count = c_p.sum()


        if action == 0 :  # 左左
            if q_c_p1[0] != 1:       # 走通会改，没有走通就不变
               max_idx1 = np.argmax(q_c_p1)
               q_c_p1[max_idx1] = 0      # 置为0
               q_c_p1[max_idx1-1] = 1    # 向左移动

               if q_c_p2[0] != 1:    # 满足前述条件执行此指令,不满足就不变
                   max_idx2 = np.argmax(q_c_p2)
                   q_c_p2[max_idx2] = 0
                   q_c_p2[max_idx2 - 1] = 1

        elif action == 1:  # 左右
            if q_c_p1[0] != 1:
               max_idx1 = np.argmax(q_c_p1)
               q_c_p1[max_idx1] = 0
               q_c_p1[max_idx1-1] = 1

               if q_c_p2[-1] != 1:
                   max_idx2 = np.argmax(q_c_p2)
                   q_c_p2[max_idx2] = 0
                   q_c_p2[max_idx2 + 1] = 1

        elif action == 2:  # 左工
            if q_c_p1[0] != 1:
               max_idx1 = np.argmax(q_c_p1)
               q_c_p1[max_idx1] = 0
               q_c_p1[max_idx1-1] = 1

               max_idx2 = np.argmax(q_c_p2)
               if c_p[max_idx2] == 1:  # 判断此处是否有箱
                   c_p[max_idx2] = 0   # 有箱就变，无箱就不变

        elif action == 3:  # 右左
            # 不需要判断  将交叉变成惩罚（看是否可行）
            max_idx1 = np.argmax(q_c_p1)
            q_c_p1[max_idx1] = 0
            q_c_p1[max_idx1 + 1] = 1

            max_idx2 = np.argmax(q_c_p2)
            q_c_p2[max_idx2] = 0
            q_c_p2[max_idx2 - 1] = 1

        elif action == 4:  # 右右
            max_idx1 = np.argmax(q_c_p1)
            q_c_p1[max_idx1] = 0
            q_c_p1[max_idx1 + 1] = 1

            if q_c_p2[-1] != 1:
              max_idx2 = np.argmax(q_c_p2)
              q_c_p2[max_idx2] = 0
              q_c_p2[max_idx2 + 1] = 1

        elif action == 5:  # 右工
            max_idx1 = np.argmax(q_c_p1)
            q_c_p1[max_idx1] = 0
            q_c_p1[max_idx1 + 1] = 1

            max_idx2 = np.argmax(q_c_p2)
            if c_p[max_idx2] == 1:
                c_p[max_idx2] = 0

        elif action == 6:  # 工左
            max_idx1 = np.argmax(q_c_p1)
            if c_p[max_idx1] == 1:
                c_p[max_idx1] = 0

            max_idx2 = np.argmax(q_c_p2)
            q_c_p2[max_idx2] = 0
            q_c_p2[max_idx2 - 1] = 1

        elif action == 7:  # 工右
            max_idx1 = np.argmax(q_c_p1)
            if c_p[max_idx1] == 1:
                c_p[max_idx1] = 0

            if q_c_p2[-1] != 1:
                max_idx2 = np.argmax(q_c_p2)
                q_c_p2[max_idx2] = 0
                q_c_p2[max_idx2 + 1] = 1

        elif action == 8:  # 工工
            max_idx1 = np.argmax(q_c_p1)
            if c_p[max_idx1] == 1:
                c_p[max_idx1] = 0

            max_idx2 = np.argmax(q_c_p2)
            if c_p[max_idx2] == 1:
                c_p[max_idx2] = 0

        # 上述所有选项选择一个后，继续执行以下语句
        self.set = np.hstack((c_p, q_c_p1, q_c_p2))
        print("剩余箱子: {}".format(c_p.sum()))
        s_ = np.hstack((c_p, q_c_p1, q_c_p2))

        if s_[0:self.c_p_len].sum() == 0:  # 所有集装箱总数
            reward = 10
            done = True

        # 最少相隔一个贝位(此处有一定的问题)
        elif abs(np.argmax(s_[self.c_p_len:self.q_c_p1_len])-np.argmax(s_[self.q_c_p1_len:self.q_c_p2_len])) <= self.constraint:
            reward = -10
            done = True

        # 假设一个联合动作只能操作一个箱
        elif s_[0:self.c_p_len].sum() < set_count:
            reward = 5
            done = False

        # 如果只单纯移动没有奖励
        else:
            reward = 0
            done = False  # 训练使用
            # done = True   # 测试代码使用


        return s_, reward, done


    # 刷新环境（仿照gym框架）
    def render(self):
        time.sleep(0.01)

# 测试代码
def update():
    for t in range(1):
        s = env.reset()  # 循环开始前需要初始化 所有内容初始化
        while True:
            env.render()
            a = 0
            s_, r, done =env.step(a)
            #print(s_)
            print(r)
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

# 列表形式最大值选取
# max_vaule2 = max(q_c_p2)
# max_idx2 = q_c_p2.index(max_vaule2)



