from quay_crane_env import Envir
from quay_crane_RL import DeepQNetwork
import matplotlib.pyplot as plt
import time
import datetime  # 记录做实验图片时间


def run_Envir():
    step = 0
    count_x = []
    count_figure = []
    episode_sum = 1000          # 迭代次数
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
    xx = int(datetime.datetime.now().strftime('%Y%m%d%H%'))
    plt.savefig('C:/Users/89317/Desktop/实验图片/{}收敛曲线.png'.format(xx), bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    time_start = time.time()  # 记录开始时间
    env = Envir()
    RL = DeepQNetwork(env.n_actions, env.n_features,
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

