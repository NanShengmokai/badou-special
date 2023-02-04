# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt
#
# # # 使用numpy生成200个随机点       #shape（200，1），min：-0.5,max：0.5
# # x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]  # np.newaxis的功能:插入新维度， #shape（200，1），min：-0.5,max：0.5
# # noise = np.random.normal(0, 0.02, x_data.shape)  # 均值，标准差，形状      shape（200，1）
# # y_data = np.square(x_data) + noise  # 返回新数组，新数组为原数组的平方
#
# x_data=np.linspace(-0.5,0.5,200)[:,np.newaxis]
# noise=np.random.normal(0,0.02,x_data.shape)
# y_data=np.square(x_data)+noise
#
# # 定义两个placeholder存放输入数据
# # x = tf.placeholder(tf.float32, [None, 1])
# # y = tf.placeholder(tf.float32, [None, 1])
# x=tf.placeholder(tf.float32,[None,1])
# y=tf.placeholder(tf.float32,[None,1])
#
# # 定义神经网络中间层
# # Weights_L1 = tf.Variable(tf.random_normal([1, 10]))  # 第一层的权重
# # biases_L1 = tf.Variable(tf.zeros([1, 10]))  # 加入偏置项
# # #tf.zeros()是常用的填充零函数
# # Wx_plus_b_L1 = tf.matmul(x, Weights_L1) + biases_L1  # w*x+b
# # L1 = tf.nn.tanh(Wx_plus_b_L1)  # 加入激活函数
# # tf.random_normal()函数用于从服从指定正太分布的数值中取出指定个数的值。
# # tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
# #     shape: 输出张量的形状，必选
# #     mean: 正态分布的均值，默认为0
# #     stddev: 正态分布的标准差，默认为1.0
# #     dtype: 输出的类型，默认为tf.float32
# #     seed: 随机数种子，是一个整数，当设置之后，每次生成的随机数都一样
# #     name: 操作的名称
# w_L1=tf.Variable(tf.random_normal([1,10]))
# b_L1=tf.Variable(tf.zeros([1,10]))
# w_x_b1=tf.matmul(x,w_L1)+b_L1
# L1=tf.nn.tanh(w_x_b1)
#
# w_L2=tf.Variable(tf.random_normal([10,1]))
# b_L2=tf.Variable(tf.zeros([1,1]))
# w_x_b2=tf.matmul(L1,w_L2)+b_L2
# L2=tf.nn.tanh(w_x_b2)
# # 定义神经网络输出层
# # Weights_L2 = tf.Variable(tf.random_normal([10, 1]))  # 第二层的权重
# # biases_L2 = tf.Variable(tf.zeros([1, 1]))  # 加入偏置项
# # Wx_plus_b_L2 = tf.matmul(L1, Weights_L2) + biases_L2  # w*x+b
# # prediction = tf.nn.tanh(Wx_plus_b_L2)  # 加入激活函数
# prediction=L2
#
# # 定义损失函数（均方差函数（mse））
# # loss = tf.reduce_mean(tf.square(y - prediction))
# loss=tf.reduce_mean(tf.square(y-prediction))
# # loss=tf.reduce_mean(tf.square(y-prediction))
# #tf.reduce_mean 函数用于计算张量tensor沿着指定的数轴（tensor的某一维度）上的的平均值，主要
# # 用作降维或者计算tensor（图像）的平均值。
# # 定义反向传播算法（使用梯度下降算法训练）
# # train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)  # 0.1:学习率，minimiz：取最小值
# # train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)
# # train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)
#
# train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)
#
# with tf.Session() as sess:
#     # 变量初始化
#     sess.run(tf.global_variables_initializer())
#     # 训练2000次
#     for i in range(2000):
#         sess.run(train_step, feed_dict={x: x_data, y: y_data})
#
#     # 获得预测值
#     prediction_value = sess.run(prediction, feed_dict={x: x_data})
#     #正常情况下推理数据和训练数据不应该是一样的（不能用x:x_data）
#
#     # 画图
#     plt.figure()
#     plt.scatter(x_data, y_data)  # 散点是真实值
#     plt.plot(x_data, prediction_value, 'r-', lw=5)  # 曲线是预测值
#     plt.show()


# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for i in range(2000):
#         sess.run(train_step,feed_dict={x:x_data,y:y_data})
#     prediction_step=sess.run(prediction,feed_dict={x:x_data})
#
#     plt.figure()
#     plt.scatter(x_data,y_data)
#     plt.plot(x_data,prediction_step,'r-',lw=5)
#     plt.show()


# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt
#
# x_data=np.linspace(-0.5,0.5,200)[:,np.newaxis]
# noise=np.random.normal(0,0.02,x_data.shape)
# y_data=np.square(x_data)+noise
#
# x=tf.placeholder(tf.float32,[None,1])
# y=tf.placeholder(tf.float32,[None,1])
# #
# w1=tf.Variable(tf.random_normal([1,10]))
# b1=tf.Variable(tf.zeros([1,10]))
# w1_x_b1=tf.matmul(x,w1)+b1
# L1=tf.nn.tanh(w1_x_b1)
# #
# w2=tf.Variable(tf.random_normal([10,1]))
# b2=tf.Variable(tf.zeros([1,1]))
# w2_x_b2=tf.matmul(L1,w2)+b2
# L2=tf.nn.tanh(w2_x_b2)
#
# prediction=L2
#
# loss=tf.reduce_mean(tf.square(y-prediction))
#
# train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#
#     for i in range(2000):
#         sess.run(train_step,feed_dict={x:x_data,y:y_data})
#
#     prediction_step=sess.run(prediction,feed_dict={x:x_data})
#
#     plt.figure()
#     plt.scatter(x_data,y_data)
#     plt.plot(x_data,prediction_step,'r',lw=5)
#     plt.show()


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

x_data=np.linspace(-0.5,0.5,300)[:,np.newaxis]
noise=np.random.normal(0,0.02,x_data.shape)
y_data=np.square(x_data)+noise

x=tf.placeholder(tf.float32,[None,1])
y=tf.placeholder(tf.float32,[None,1])

w1=tf.Variable(tf.random_normal([1,10]))
b1=tf.Variable(tf.zeros([1,10]))
w1_x_b1=tf.matmul(x,w1)+b1
L1=tf.nn.tanh(w1_x_b1)

w2=tf.Variable(tf.random_normal([10,1]))
b2=tf.Variable(tf.zeros([1,1]))
w2_x_b2=tf.matmul(L1,w2)+b2
L2=tf.nn.tanh(w2_x_b2)

prediction=L2

loss=tf.reduce_mean(tf.square(y-prediction))

train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(5000):
        sess.run(train_step,feed_dict={x:x_data,y:y_data})

    prediction_step=sess.run(prediction,feed_dict={x:x_data})

    plt.figure()
    plt.scatter(x_data,y_data)
    plt.plot(x_data,prediction_step,'r',lw=5)
    plt.show()