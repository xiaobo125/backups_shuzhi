from quay_crane_env_q import Envir            # 从迷宫文件中中导入类
from RL_brain import QLearningTable   # 从主干部分导入Q表


def update():
	for episode in range(150):
		observation = env.reset()  # 环境重置
		print(episode)
		while True:
			env.render()           # 环境刷新
			action = RL.choose_action(str(observation))  # 基于观测值挑选动作
			# print("observation: {}".format(observation))
			observation_, reward, done = env.step(action)
			RL.learn(str(observation), action, reward, str(observation_))  # 进行一个transition学习
			# print(RL.q_table)
			observation = observation_
			if done:    # 如果完成就是跳出
				break
	print('game over')


if __name__ == '__main__':
	env = Envir()                # 建立整体环境  实例化对env象，里面带self
	# print("env.n_actions: {}".format(env.n_actions))
	RL = QLearningTable(actions=list(range(env.n_actions)))  # 将四个动作直接编码，动作编码为列表值
	update()
