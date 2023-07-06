import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x.pow(3)+0.1*torch.randn(x.size())   # 加随机噪声增加数据复杂度
# pow函数是幂次方，randn生成维度相同的（0，1]随机数字

x, y = (Variable(x), Variable(y))  # 数据转化类型，tensor不能反向船传播，variable可以反向传播

# plt.scatter(x,y)
# plt.scatter(x.data,y.data)
# plt.scatter(x.data.numpy(),y.data.numpy())
# plt.show()

# 神经网络一般模板
class Net(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):  # 实例化对象直接执行
        super(Net, self).__init__()  # 使用父类的初始化来初始化子类
        self.hidden1 = nn.Linear(n_input, n_hidden)
        self.hidden2 = nn.Linear(n_hidden, n_hidden)
        self.predict = nn.Linear(n_hidden, n_output)


    def forward(self, input):  # 前向传播
        out = self.hidden1(input)
        out = torch.relu(out)
        out = self.hidden2(out)
        out = torch.sigmoid(out)
        out = self.predict(out)

        return out


net = Net(1, 20, 1)
print(net)

optimizer = torch.optim.SGD(net.parameters(), lr=0.1)  # 随机梯度下降策略
loss_func = torch.nn.MSELoss()  # 损失函数使用据均方损失函数，预测值和真实值之差的平方和的平均值

for t in range(10000):
    prediction = net(x)
    loss = loss_func(prediction, y)
    optimizer.zero_grad()   # 梯度置为0，因为是个累加，所以每个过程需要重置为0
    loss.backward()  # 损失反向传播
    optimizer.step()  # 梯度优化

    if t % 2500 == 0:
        plt.cla()  # 清除坐标轴
        plt.scatter(x.data.numpy(), y.data.numpy())   # 取其中的数据绘制散点图
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)  # 绘制预测红线的曲线图  lw是线宽
        plt.text(0.5, 0, 'Loss = %.4f' % loss.data, fontdict={'size': 20, 'color': 'red'})
        # 上句分别是横纵坐标，损失值保留四位小数，文字大小和颜色
        plt.pause(0.05)  # 停顿五秒

plt.show()  # 画图展示
