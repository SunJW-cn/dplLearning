import random
import torch
from d2l import torch as d2l
from d2l.torch import linreg
from torch.onnx.symbolic_opset9 import tensor


def synthetic_data(w,b,num_examples):
    """生成y=Xw+b+噪声"""
    X = torch.normal(0,1,(num_examples,len(w)))
    '''通过（0，1）均值方差生成一个x张量，大小等于w的大小'''
    y = torch.matmul(X,w)+b
    y +=torch.normal(0,0.1,y.shape)
    '''生成噪声部分'''
    return X,y.reshape((-1,1))
    '''reshape(-1,1)将y变成列向量'''

true_w = torch.tensor([2,-3.4])
true_b = 4.2
features,labels = synthetic_data(true_w,true_b,1000)
'''
通过函数生成1000个样本的数据集，这个数据集是根据w,b的真实值生成的
'''
d2l.set_figsize()
d2l.plt.scatter(features[:,1].detach().numpy(),labels.detach().numpy(),1)
d2l.plt.show()
'''
显示数据集的分布
'''

def data_iter(batch_size,features,labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    #生成一个0到num_examples-1的列表即代表所有数据的下标，我们通过indices来标识每个样本
    random.shuffle(indices)
    #将所有样本打乱
    for i in range(0,num_examples,batch_size):
    #从零开始，到examples结束，步长为batch_size
        batch_indices = torch.tensor(indices[i:min(i+batch_size,num_examples)])
        #将indices中的数据取出来，取到min(i+batch_size,num_examples)为止,即取到i+batch_size和num_examples-1中较小的那个
        yield features[batch_indices],labels[batch_indices]
    #生成一个迭代器，每次返回一个batch的数据,这个迭代器会记录最近一次生成的位置

batch_size = 10
w = torch.normal(0,0.01,size=(2,1),requires_grad=True)
b = torch.zeros(1,requires_grad =True)
#设置初始w,b

def linreg(X,w,b):
    return torch.matual(X,w) + b

def square_loss(y_hat,y):
    return (y_hat-y.reshape(y_hat.shape))**2/2

def sgd(params,lr,batch_size):
    with torch.no_gard():
        #更新参数时，无需将梯度改变记录
        for param in params:
            param -= lr*param.gard()/batch_size
            param.gard.zero_()

lr = 0.03
num_epochs = 3
net = linreg
loss = square_loss
