import torch
from torch.distributions import multinomial
from d2l import torch as d2l

fair_probs = torch.ones(6) / 6
multinomial.Multinomial(1, fair_probs).sample()
'''
生成概率向量，每个概率设置为均匀的1/6
第一个参数为取样次数，第二个参数是指定概率向量，通过.simple从分布中抽取一个样本
在本例中，假设抽样结果为tensor([0,0,0,1,0,0,])表示在索引3处，出现一次
即代表着这次抽样掷骰子掷到了4
'''

'''
tensor([0.1540, 0.1700, 0.1840, 0.1700, 0.1610, 0.1610])
1000次结果表明大约是符合0.167的概率
'''
counts = multinomial.Multinomial(10, fair_probs).sample((500,))
'''
通过sample((500,))抽取出500个样本，存放在一个二维数组中，每一行代表一个样本可以在第二个参数指定列数，这里默认
'''
cum_counts = counts.cumsum(dim=0)
'''
通过cumsum来计算指定维度的累积和，指定维度为0，则按行计算累积和
counts.cumsum(dim=0)是一个新的张量，每一行计算从counts的第一行到该行对应位置的累积和
即counts每一行都是之前所有行的数值相加
'''
estimates = cum_counts/ cum_counts.sum(dim=1, keepdims=True)
'''
相似的，这里分母是计算沿列进行计算累积和，且通过keepdims保持结果维度与输入相同
将上面按行累积计数的张量中每个元素除以该行所有元素之和
将累积技术转换为概率估计，使得每一行和为1，类似于归一化处理
这样随着抽样次数增加，即行数相加，每一行的对应元素数值都越发平均
，除以行累积和后，可以体现出随时间或者随抽样次数增加概率的变化
'''
d2l.set_figsize((10,15))
for i in range(6):
    d2l.plt.plot(estimates[:,i].numpy(),label = ("P(die="+str(i+1)+")"))
d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Groups of experiments')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend();
d2l.plt.show()