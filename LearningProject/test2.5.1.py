import torch
from torch.nn.init import xavier_uniform

x = torch.arange(4.0)
x.requires_grad_(True)
'''
等价于x = torch.arange(4.0,requires_grad=True)
requires_grad(True)表示需要计算梯度,
Pytorch会自动跟踪所有在计算图中对张量的操作,并使用自动微分算法计算梯度
'''
x.grad
'''
该参数默认是none，第一次为张量自身计算梯度，
调用backward时，该属性会变成一个tensor张量类型，
该属性将会包含计算所得的梯度，再次调用backward时
将会再次基础上累加，所以下次使用可能需要清零
'''

y = 2 * torch.dot(x, x)
'''
对x进行点积操作，结果是一个标量即一个数，结果为
tensor(28., grad_fn=<MulBackward0>)
表示这个张量是由一个乘法操作生成的，并且这个操作会计算梯度
'''
y.backward()
# 调用backward函数，计算梯度，并赋值给x.grad
x.grad
print(x.grad)
print(x.grad == 4 * x)
# 清除累积的梯度
x.grad.zero_()
y = x.sum()
print(y)
y.backward()
x.grad
'''
前面的y=2*dot(x,x)其实应表示为y= 2(x1*x1+x2*x2+x3*x3+x4*x4)
所以对y反向传播求梯度时，是对x1,x2,x3,x4分别求偏导，所以x.grad=4*x
因此，此处y=sum(x)其实应为y=x1+x2+x3+x4，
对x1,x2,x3,x4分别求偏导，所以x.grad=1
'''
print(x.grad)

x.grad.zero_()
y = x * x
'''
等价于y.backward(torch.ones(len(x)))
对于非标量调用backward需要传入一个gradient参数，
作用是指定微分函数关于self的梯度,该参数应该是一个和self同形的张量
这里求偏导数之和，传递梯度为1
采用y.sum的原因是如果不用y.sum的话，则需要这样写：
y=x*x
gradient=torch.ones_like(y)
y.backward(gradient)
y是对x做一个元素级的乘法，即此时y=tensor([0., 1., 4., 9.])
'''
y.sum().backward()
'''
对y张量进行所有元素进行求和操作，结果为一个标量14
执行反向传播，计算y.sum对于x中每个元素的偏导数
y=x^2,则y'=2x,对于y.sum(),将每个元素梯度相加，这样总梯度是每个元素偏导的总和
y.sum()=y1+y2+y3+y4,所以y.sum()对于y的梯度均为1
y=xi*xi,所以y对于x的梯度为2xi,
所以y.sum()对于x的梯度为2x1+2x2+2x3+2x4=2x.sum()
'''
x.grad
print(x.grad)

'''
分离计算，即z是y与x的函数，y又是x的函数，有时需要将y视作常数，只考虑y
我们分离y返回一个u,u的值和y相同，梯度不会经由u流向x
'''
x.grad.zero_()
y = x*x
u = y.detach()
z = u*x

z.sum().backward()
print(x.grad)
#结果为tensor([0., 1., 4., 9.])，证明此时z对x的偏导为u,而不是3x^2
print(u)
#结果为tensor([0., 1., 4., 9.])
print(x.grad == u)

x.grad.zero_()
y.sum().backward()
'''
同样的这里y也只是x的直接相乘，无法直接反向，选择sum后再反向
'''
print(x.grad == 2*x)


