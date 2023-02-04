# # PyTorch 的设计遵循tensor→variable(autograd)→nn.Module 三个由低到高的抽象层次，
# # 分别代表高维数组（张量）、自动求导（变量）和神经网络（层/模块）
# import torch
# import torch.nn as nn    #为最大化灵活性而设计，与 autograd 深度整合的神经网络库；
# import torch.optim as optim   #与torch.nn 一起使用的优化包，包含 SGD、RMSProp、LBFGS、Adam 等标准优化方式；
# import torch.nn.functional as F
# import torchvision
# import torchvision.transforms as transforms#构建计算机视觉模型。
#
# # torchvision是pytorch的一个图形库，它服务于PyTorch深度学习框架的，主要用来构建计算机视觉模型。
# # torchvision.transforms主要是用于常见的一些图形变换。以下是torchvision的构成：
#
# # 1.torch ：类似 NumPy 的张量库，支持GPU；
# # 2.torch.autograd ：基于 type 的自动区别库，支持 torch 之中的所有可区分张量运行；
# # 3.torch.nn ：为最大化灵活性而设计，与 autograd 深度整合的神经网络库；
# # 4.torch.optim：与 torch.nn 一起使用的优化包，包含 SGD、RMSProp、LBFGS、Adam 等标准优化方式；
# # 5.torch.multiprocessing： python 多进程并发，进程之间 torch Tensors 的内存共享；
# # 6.torch.utils：数据载入器。具有训练器和其他便利功能；
# # 7.torch.legacy(.nn/.optim) ：出于向后兼容性考虑，从 Torch 移植来的 legacy 代码；
#
# # Pytorch中，nn与nn.functional有哪些区别？
# # 相同之处：
# # 两者都继承于nn.Module
# # nn.x与nn.functional.x的实际功能相同，比如nn.Conv3d和nn.functional.conv3d都是进行3d卷积
# # 运行效率几乎相同
# # 不同之处：
# # nn.x是nn.functional.x的类封装，nn.functional.x是具体的函数接口
# # nn.x除了具有nn.functional.x功能之外，还具有nn.Module相关的属性和方法，比如：train(),eval()等
# # nn.functional.x直接传入参数调用，nn.x需要先实例化再传参调用
# # nn.x能很好的与nn.Sequential结合使用，而nn.functional.x无法与nn.Sequential结合使用
# # nn.x不需要自定义和管理weight,而nn.functional.x需自定义weight,作为传入的参数
#
# #
# class Model:
#     def __init__(self, net, cost, optimist):        #cost：损失函数，optimist：优化项
#         self.net = net  #net = MnistNet()
#         self.cost = self.create_cost(cost)
#         self.optimizer = self.create_optimizer(optimist)
#         pass
#
#     def create_cost(self, cost):        #提供俩种损失函数
#         support_cost = {
#             'CROSS_ENTROPY': nn.CrossEntropyLoss(),     #交叉熵
#             'MSE': nn.MSELoss()                         #mse
#         }
#         #支持的损失函数有俩种选择：交叉熵，MSE
#         return support_cost[cost]
#
#     def create_optimizer(self, optimist, **rests):      #支持的优化项，提供了三种
#         support_optim = {
#             'SGD': optim.SGD(self.net.parameters(), lr=0.1, **rests),
#             'ADAM': optim.Adam(self.net.parameters(), lr=0.01, **rests),
#             'RMSP':optim.RMSprop(self.net.parameters(), lr=0.001, **rests)
#         }
#
#         return support_optim[optimist]
#     #torch.optim.SGDtorch.optim.SGD(params, lr=<required parameter>, momentum=0, dampening=0,
#     # weight_decay=0, nesterov=False)：随机梯度下降
#     #虽然叫做“随机梯度下降”，但是本质上还是还是实现的批量梯度下降，即用全部样本梯度的均值更新可学习参数。
#     #这里所说的全部样本可以是全部数据集，也可以是一个batch，为什么这么说？因为计算梯度是调用backward函
#     #数计算的，而backward函数又是通过损失值张量调用的，损失值的计算和样本集的选取息息相关。如果每次都使用全
#     #部样本计算损失值，那么很显然调用SGD时也就是用全部的样本梯度的均值去更新可学习参数，如果每次使用一个
#     #batch的样本计算损失值，再调用backward，那么调用SGD时就会用这个batch的梯度均值更新可学习参数。
#     #params：要训练的参数，一般我们传入的都是model.parameters()。
#     #lr：learning_rate学习率，会梯度下降的应该都知道学习率吧，也就是步长。
#     #weight_decay（权重衰退）
#     #weight_decay是在L2正则化理论中出现的概念。
#
#     def train(self, train_loader, epoches=3):       #训练函数
#         for epoch in range(epoches):
#             running_loss = 0.0      #初始化
#             for i, data in enumerate(train_loader, 0):      #train_loader：mnist-load进来的训练函数
#                 #for index,value in enumerate(names):
#                 inputs, labels = data       #data：是枚举
#
#                 self.optimizer.zero_grad()      #梯度的计算
#                 # forward + backward + optimize
#                 outputs = self.net(inputs)          #net=MnistNet（），传入之前已经在开头初始化好了,inputs=data
#                 loss = self.cost(outputs, labels)   #cost:交叉熵，开头已经定义好,labels=data
#                 loss.backward()                     #直接进行反向传播
#                 self.optimizer.step()               #进行优化(可有可无)，训练结束，是求训练误差
#
#                 running_loss += loss.item()#    计算训练误差
#
#                 # 1.item（）取出张量具体位置的元素元素值
#                 # 2.并且返回的是该位置元素值的高精度值
#                 # 3.保持原元素类型不变；必须指定位置
#                 if i % 100 == 0:
#                 #i代表当前第几轮循环，i % 100用于每隔100轮循环打印一次
#                     print('[epoch %d, %.2f%%] loss: %.3f' %
#                           (epoch + 1, (i + 1)*1./len(train_loader), running_loss / 100))
#                     running_loss = 0.0
#                 #[epoch 1, 0.00%] loss: 0.023
#
#         print('Finished Training')
#
#     def evaluate(self, test_loader):    #推理函数
#         print('Evaluating ...')     #推理
#         correct = 0 #correct是模型预测正确的图像数
#         total = 0   #本次迭代中图像的总数，
#         with torch.no_grad():  # no grad when test and predict
#         #torch.no_grad() 一般用于神经网络的推理阶段, 表示张量的计算过程中无需计算梯度
#
#             for data in test_loader:
#                 images, labels = data
#
#                 outputs = self.net(images)      #其实到这就推理完事了
#                 predicted = torch.argmax(outputs, 1)        #argmax：排序
#                 #argmax函数：torch.argmax(input, dim=None, keepdim=False)返回指定维度最大
#                 # 值的序号，dim给定的定义是：the demention to reduce，就是把dim这个维度，变成
#                 # 这个维度的最大值的index。
#
#                 #1:dim表示不同维度。特别的在dim = 0表示二维矩阵中的列，dim = 1在二维矩阵中的行。
#                 # 广泛的来说，我们不管一个矩阵是几维的，比如一个矩阵维度如下：(d0, d1,…, dn−1) ，
#                 # 那么dim = 0就表示对应到d0也就是第一个维度，dim = 1表示对应到也就是第二个维度，以此类推。
#
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item() #.item():取出具体的值
#                 #total是本次迭代中图像的总数，correct是模型预测正确的图像数，这两个可用于计算准确率
#
#
#         print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
#
# def mnist_load_data():
#     transform = transforms.Compose(
#         [transforms.ToTensor(),
#          transforms.Normalize([0,], [1,])])
#         #Compose里面的参数实际上就是个列表，而这个列表里面的元素就是你想要执行的transform操作。
#         #transforms.ToTensor()可以将PIL和numpy格式的数据从[0,255]范围转换到[0,1] ，具体做
#         #法其实就是将原始数据除以255。另外原始数据的shape是（H x W x C），通过transforms.ToTensor()
#         #后shape会变为（C x H x W）
#         #transforms.Normalize:(进行正态分布操作)这个函数的输出output[channel] = (input[channel] - mean[channel]) / std[channel]。
#         #这里[channel]的意思是指对特征图的每个通道都进行这样的操作。【mean为均值，std为标准差】
#         #。transforms.Normalize用于数据的归一化，[0,], [1,]用于设置参数mean与std，
#
#
#     #torchvision是pytorch的一个图形库，它服务于PyTorch深度学习框架的，主要用来构建计算机视觉模型。
#     # torchvision.transforms主要是用于常见的一些图形变换。以下是torchvision的构成：
#
#     #1:torchvision.datasets: 一些加载数据的函数及常用的数据集接口；
#     #2:torchvision.models: 包含常用的模型结构（含预训练模型），例如AlexNet、VGG、ResNet等；
#     #3:torchvision.transforms: 常用的图片变换，例如裁剪、旋转等；
#     #4:torchvision.utils: 其他的一些有用的方法。
#     #torchvision.transforms.Compose()这个类的主要作用是串联多个图片变换的操作
#
#
#     trainset = torchvision.datasets.MNIST(root='./data', train=True,
#                                             download=True, transform=transform)
#     trainset=torchvision.datasets.MNIST(root='./data',train=True,download=True,transform=transform)
#     # CLASS torchvision.datasets.MNIST(root: str, train: bool = True,
#     # transform: Optional[Callable] = None, target_transform:Optional[Callable] = None,
#     # download: bool = False)
#     # root(string)： 表示数据集的根目录，其中根目录存在MNIST / processed / training.pt和MNIST / processed / test.pt的子目录
#     # train(bool, optional)： 如果为True，则从training.pt创建数据集，否则从test.pt创建数据集
#     # download(bool, optional)： 如果为True，则从internet下载数据集并将其放入根目录。如果数据集已下载，则不会再次下载
#     # transform(callable, optional)： 接收PIL图片并返回转换后版本图片的转换函数
#     # target_transform(callable, optional)： 接收PIL接收目标并对其进行变换的转换函数
#
#     trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
#                                               shuffle=True, num_workers=2)
#     trainloader=torch.utils.data.DataLoader(trainset,batch_size=32,shuffle=True,num_workers=2)
#     #DataLoader是PyTorch中的一种数据类型。PyTorch中数据读取的一个重要接口是torch.utils.data.DataLoader，
#     # 该接口定义在dataloader.py脚本中，只要是用PyTorch来训练模型基本都会用到该接口，该接口主要用
#     # 来将自定义的数据读取接口的输出或者PyTorch已有的数据读取接口的输入按照batch size封装成Tensor。
#     # （方便产生一个可迭代对象(iterator)，每次输出指定batch_size大小的Tensor）
#     #可以设置参数：
#     #dataset：包含所有数据的数据集
#     #batch_size: 每一小组所包含数据的数量
#     #Shuffle: 是否打乱数据位置，当为Ture时打乱数据，全部抛出数据后再次dataloader时重新打乱。
#     #sampler: 自定义从数据集中采样的策略，如果制定了采样策略，shuffle则必须为False.
#     #Batch_sampler: 和sampler一样，但是每次返回一组的索引，和batch_size, shuffle, sampler, drop_last互斥。
#     #num_workers: 使用线程的数量，当为0时数据直接加载到主程序，默认为0。
#     #collate_fn:是用来处理不同情况下的输入dataset的封装，一般采用默认即可，除非你自定义的数据读取输出非常少见
#     #pin_memory: s是一个布尔类型，为T时将会把数据在返回前先复制到CUDA的固定内存中
#     #drop_last: 布尔类型，为T时将会把最后不足batch_size的数据丢掉，为F将会把剩余的数据作为最后一小组。
#     #timeout：默认为0。当为正数的时候，这个数值为时间上限，每次取一个batch超过这个值的时候会报错。此参数必须为正数。
#     #worker_init_fn: 和进程有关系，暂时用不到
#
#
#
#     testset = torchvision.datasets.MNIST(root='./data', train=False,
#                                            download=True, transform=transform)
#     testloader = torch.utils.data.DataLoader(testset, batch_size=32,shuffle=True, num_workers=2)
#     return trainloader, testloader
#
# class MnistNet(torch.nn.Module):    #构建的网络结构
#     def __init__(self):     #需要训练的数据，#需要训练的层在__init__中
#         super(MnistNet, self).__init__()
#         #super(Net, self).__init__()
#         # Python中的super(Net, self).__init__()是指首先找到Net的父类（比如是类NNet），然后把类
#         # Net的对象self转换为类NNet的对象，然后“被转换”的类NNet对象调用自己的init函数，
#         # 其实简单理解就是子类把父类的__init__()放到自己的__init__()当中，这样子类就有了父类的
#         # __init__()的那些东西。
#         #回过头来看看我们的我们最上面的代码，Net类继承nn.Module，super(Net, self).__init__()就是对
#         #继承自父类nn.Module的属性进行初始化。而且是用nn.Module的初始化方法来初始化继承的属性。
#         self.fc1 = torch.nn.Linear(28 * 28, 512)
#         self.fc2 = torch.nn.Linear(512, 512)    #中间层
#         self.fc3 = torch.nn.Linear(512, 10)     #输出层
#
#     def forward(self, x):   #不需要训练的数据
#         x = x.view(-1, 28*28)       #reshape。-1：一维
#         x = F.relu(self.fc1(x))     #F：functional
#         x = F.relu(self.fc2(x))     #import torch.nn.functional as F
#         x = F.softmax(self.fc3(x), dim=1)
#         return x
#
# class MnistNet(torch.nn.Module):
#     def __init__(self):
#         super(MnistNet, self).__init__()
#         self.fc1=torch.nn.Linear(28*28,512)
#         self.fc2=torch.nn.Linear(512,512)
#         self.fc3=torch.nn.Linear(512,10)
#     def forward(self,x):
#         x=x.view(-1,28*28)
#         x=F.relu(self.fc1(x))
#         x=F.relu(self.fc2(x))
#         x=F.softmax(self.fc3(x),dim=1)
#         #argmax函数：torch.argmax(input, dim=None, keepdim=False)返回指定维度最大
#         #值的序号，dim给定的定义是：the demention to reduce，就是把dim这个维度，变成
#         #这个维度的最大值的index。
#         #1:dim表示不同维度。特别的在dim = 0表示二维矩阵中的列，dim = 1在二维矩阵中的行。
#         return x
#
# if __name__ == '__main__':
#     # train for mnist
#     net = MnistNet()        #构建网络结构
#     model = Model(net, 'CROSS_ENTROPY', 'RMSP')     #（net，损失函数，优化项）
#     train_loader, test_loader = mnist_load_data()
#     model.train(train_loader)           #训练
#     model.evaluate(test_loader)         #推理

import torch.utils as utils
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


class Model:
    def __init__(self, net, cost, optimist):
        self.net = net
        self.cost = self.create_cost(cost)
        self.optimizer = self.create_optimizer(optimist)
        pass

    def create_cost(self, cost):
        support_cost = {
            'CROSS_ENTROPY': nn.CrossEntropyLoss(),
            'MSE': nn.MSELoss()
        }
        return support_cost[cost]

    def create_optimizer(self, optimist, **rests):
        support_optim = {
            'SGD': optim.SGD(self.net.parameters(), lr=0.1, **rests),
            'ADAM': optim.Adam(self.net.parameters(), lr=0.01, **rests),
            'RMSP': optim.RMSprop(self.net.parameters(), lr=0.001, **rests)
        }
        return support_optim[optimist]

    def train(self, train_loader, epoches=3):
        for epoch in range(epoches):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self.cost(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if i % 100 == 0:
                    print('[epoch %d ,%.2f%%] loss : %.3f' % (
                    epoch + 1, (i + 1) * 1. / len(train_loader), running_loss / 100))
                    running_loss = 0.0
        print('Finished Training')

    def evaluate(self, test_loader):
        print('Evaluating...')
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                outputs = self.net(images)
                predicted = torch.argmax(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        # print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
        print('...:%d %%' % (100 * correct / total))


def mnist_L_Data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0, ], [1, ])])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=2)
    return trainloader, testloader


class MnistNet(torch.nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x


if __name__ == '__main__':
    net = MnistNet()
    model = Model(net, 'CROSS_ENTROPY', 'RMSP')
    train_loader, test_loader = mnist_L_Data()
    model.train(train_loader)
    model.evaluate(test_loader)