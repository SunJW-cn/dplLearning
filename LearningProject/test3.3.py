import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
from torch import nn

true_w = torch.tensor([2, -3.14])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)


# 仍然通过人工生成f和l

def load_array(data_arrays, batch_size, is_train=True):  # @save
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


batch_size = 10
data_iter = load_array((features, labels), batch_size)

next(iter(data_iter))

net = nn.Sequential(nn.Linear(2, 1))
# 定义一个线性回归模型，这里为全连接层，输入维度为2，输出维度为1
net[0].weight.data.normal_(0, 0.01)
# 初始化权重参数,和3.2中通过均值分布手动生成是一样的，设置均值为零，方差为1
net[0].bias.data.fill_(0)
# 初始化偏置参数，均值为零
loss = nn.MSELoss()
# 定义损失函数，使用MSELoss类，被称为平方/L2范数，默认返回所有样本损失的均值
trainer = torch.optim.SGD(net.parameters(), lr=0.03)
# 定义优化器，使用SGD类，通过net.parameters()从模型中获取指定优化参数，
# 小批量随机梯度下降只需要设置lr,这里设置学习率为0.03
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

# 训练完毕后我们比较一下真实参数和模型参数
w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)
