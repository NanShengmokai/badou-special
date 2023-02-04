# #该文件的目的是构造神经网络的整体结构，并进行训练和测试（评估）过程
# import tensorflow as tf
# import numpy as np
# import time
# import math
# import Cifar10_data
#
#
# max_steps=4000
# batch_size=100
# num_examples_for_eval=10000
# data_dir="Cifar_data/cifar-10-batches-bin"          #数据的目录
#
# #创建一个variable_with_weight_loss()函数，该函数的作用是：
# #   1.使用参数w1控制L2 loss的大小
# #   2.使用函数tf.nn.l2_loss()计算权重L2 loss
# #   3.使用函数tf.multiply()计算权重L2 loss与w1的乘积，并赋值给weights_loss
# #   4.使用函数tf.add_to_collection()将最终的结果放在名为losses的集合里面，方便后面计算神经网络的总体loss，
# def variable_with_weight_loss(shape,stddev,w1):     #权重控制函数
#     var=tf.Variable(tf.truncated_normal(shape,stddev=stddev))
#     #tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None) :
#     #shape表示生成张量的维度（a * a），mean是均值，stddev是标准差,这个函数产生正态分布，均值和标准差自
#     #己设定。这是一个截断的产生正态分布的函数，生成的值服从具有指定平均值和标准偏差的正态分布，换句话说，
#     #产生的值如果与均值的差值大于两倍的标准差则丢弃重新选择。和一般的正态分布的产生随机数据比起来，这个函数
#     #产生的随机数与均值的差距不会超过两倍的标准差，但是一般的别的函数是可能的
#     #X落在（μ-3σ，μ+3σ）以外的概率小于千分之三，在实际问题中常认为相应的事件是不会发生的，
#     #在tf.truncated_normal中如果x的取值在区间（μ-2σ，μ+2σ）之外则重新进行选择。这样保证了生成的值都在均值附近。
#     if w1 is not None:
#         weights_loss=tf.multiply(tf.nn.l2_loss(var),w1,name="weights_loss")
#         #tf.nn.l2_loss(t, name=None)正则化函数：这个函数的作用是利用 L2 范数来计算张量的误差值，但是没有开方并
#         # 且只取L2 范数的值的一半，具体如下：output = sum(t ** 2) / 2
#         tf.add_to_collection("losses",weights_loss)
#     return var
# # def variable_with_weight_loss(shape,stddev,w1):
# #     var=tf.Variable(tf.truncated_normal(shape,stddev=stddev))
# #     if w1 is not None:
# #         wight_loss=tf.multiply(tf.nn.l2_loss(var),w1,name="weights_loss")
# #         tf.add_to_collection("losses",wight_loss)
#
#
# #使用上一个文件里面已经定义好的文件序列读取函数读取训练数据文件和测试数据从文件.
# #其中训练数据文件进行数据增强处理，测试数据文件不进行数据增强处理
# images_train,labels_train=Cifar10_data.inputs(data_dir=data_dir,batch_size=batch_size,distorted=True)   #distorted：图像增强
# images_test,labels_test=Cifar10_data.inputs(data_dir=data_dir,batch_size=batch_size,distorted=None)
#
#
# #创建x和y_两个placeholder，用于在训练或评估时提供输入的数据和对应的标签值。
# #要注意的是，由于以后定义全连接网络的时候用到了batch_size，所以x中，第一个参数不应该是None，而应该是batch_size
# x=tf.placeholder(tf.float32,[batch_size,24,24,3])
# y_=tf.placeholder(tf.int32,[batch_size])        #y_:输出值
#
# # #创建第一个卷积层 shape=(kh,kw,ci,co)
# # kernel1=variable_with_weight_loss(shape=[5,5,3,64],stddev=5e-2,w1=0.0)
# # # kernel：H,W：5*5,3，输入通道数，64：输出通道数
# # conv1=tf.nn.conv2d(x,kernel1,[1,1,1,1],padding="SAME")      #x：输入，kernel1：权重，[1,1,1,1]:步长，padding:模式，
# # bias1=tf.Variable(tf.constant(0.0,shape=[64]))
# # relu1=tf.nn.relu(tf.nn.bias_add(conv1,bias1)) #conv1+bias1
# # pool1=tf.nn.max_pool(relu1,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME")
# # #ksize： (batch_size, height, width, channels)
# # #k_size : 池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，因为我们不想在batch和channels上做池化，所以这两个维度设为了1
# # #strides : 窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1]
# # #padding： 填充的方法，SAME或VALID，SAME表示添加全0填充，VALID表示不添加
# # # [1, 2, 2, 1]图像如下
# # # [[□  [□
# # #   □]  □]]
# kernel1=variable_with_weight_loss(shape=[5,5,3,64],stddev=5e-2,w1=0.0)
# conv1=tf.nn.conv2d(x,kernel1,[1,1,1,1],padding="SAME")
# bias1=tf.Variable(tf.constant(0.0,shape=[64]))
# relu1=tf.nn.relu(tf.nn.bias_add(conv1,bias1))
# pool1=tf.nn.max_pool(relu1,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME")
#
#
# #创建第二个卷积层
# # kernel2=variable_with_weight_loss(shape=[5,5,64,64],stddev=5e-2,w1=0.0)
# # #kernel：5*5，输入通道数：64，输出通道数：64
# # conv2=tf.nn.conv2d(pool1,kernel2,[1,1,1,1],padding="SAME")
# # bias2=tf.Variable(tf.constant(0.1,shape=[64]))
# # relu2=tf.nn.relu(tf.nn.bias_add(conv2,bias2))
# # pool2=tf.nn.max_pool(relu2,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME")
#
# kernel2=variable_with_weight_loss(shape=[5,5,64,64],stddev=5e-2,w1=0.0)
# conv2=tf.nn.conv2d(pool1,kernel2,[1,1,1,1],padding="SAME")
# bias2=tf.Variable(tf.constant(0.1,shape=[64]))
# relu2=tf.nn.relu(tf.nn.bias_add(conv2,bias2))
# pool2=tf.nn.max_pool(relu2,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME")
#
# # #因为要进行全连接层的操作，所以这里使用tf.reshape()函数将pool2输出变成一维向量，并使用get_shape()函数获取扁平化之后的长度
# # reshape=tf.reshape(pool2,[batch_size,-1])    #这里面的-1代表将pool2的三维结构拉直为一维结构
# # dim=reshape.get_shape()[1].value             #get_shape()[1].value表示获取reshape之后的第二个维度的值
#
# reshape=tf.reshape(pool2,[batch_size,-1])
# dim=reshape.get_shape()[1].value
#
#
#
# # #建立第一个全连接层
# # weight1=variable_with_weight_loss(shape=[dim,384],stddev=0.04,w1=0.004)
# # fc_bias1=tf.Variable(tf.constant(0.1,shape=[384]))
# # fc_1=tf.nn.relu(tf.matmul(reshape,weight1)+fc_bias1)
# weight1=variable_with_weight_loss(shape=[dim,384],stddev=0.04,w1=0.004)
# fc_bias1=tf.Variable(tf.constant(0.1,shape=[384]))
# fc_1=tf.nn.relu(tf.matmul(reshape,weight1)+fc_bias1)    #w*x+b,然后进行relu
#
# #建立第二个全连接层
# # weight2=variable_with_weight_loss(shape=[384,192],stddev=0.04,w1=0.004)
# # fc_bias2=tf.Variable(tf.constant(0.1,shape=[192]))
# # local4=tf.nn.relu(tf.matmul(fc_1,weight2)+fc_bias2)
#
# weight2=variable_with_weight_loss(shape=[384,192],stddev=0.04,w1=0.004)
# fc_bias2=tf.Variable(tf.constant(0.1,shape=[192]))
# fc_2=tf.nn.relu(tf.matmul(fc_1,weight2)+fc_bias2)
#
# #建立第三个全连接层
# # weight3=variable_with_weight_loss(shape=[192,10],stddev=1 / 192.0,w1=0.0)
# # fc_bias3=tf.Variable(tf.constant(0.1,shape=[10]))
# # result=tf.add(tf.matmul(local4,weight3),fc_bias3)
#
# weight3=variable_with_weight_loss(shape=[192,10],stddev=1/192.0,w1=0.0)
# fc_bias3=tf.Variable(tf.constant(0.1,shape=[10]))
# result=tf.nn.relu(tf.matmul(fc_2,weight3)+fc_bias3)
#
#
# # #计算损失，包括权重参数的正则化损失和交叉熵损失
# # cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result,labels=tf.cast(y_,tf.int64))
# # weights_with_l2_loss=tf.add_n(tf.get_collection("losses"))    #权重参数的正则化损失
# # loss=tf.reduce_mean(cross_entropy)+weights_with_l2_loss       #交叉熵损失
# # train_op=tf.train.AdamOptimizer(1e-3).minimize(loss)  #是一个寻找全局最优点的优化算法
#
# cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result,labels=tf.cast(y_,tf.int64))
# # cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result,labels=tf.cast(y_,tf.int64))
# #tf.cast()函数将这一个标签转换成int64的数值形式
# #tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name=None)
# #参数：logits:神经网络输出层的输出，shape为[batch_size，num_classes],
# #labels:一个一维的向量,长度等于batch_size
# #返回值：返回值为长度batch_size的1D Tensor,类型和 logits 一样，值为softmax 交叉熵损失
#
# weights_with_l2_loss=tf.add_n(tf.get_collection("losses"))
# # weights_with_l2_loss=tf.add_n(tf.get_collection("losses"))
# #tf.add_n([p1, p2, p3…])函数是实现一个列表的元素的相加。就是输入的对象是一个列表，列表里的元素可以是向量，矩阵，等
# #tf.get_collection(key,scope=None)
# #该函数可以用来获取key集合中的所有元素，返回一个列表。列表的顺序依变量放入集合中的先后而定。
# #scope为可选参数，表示的是名称空间（名称域），如果指定，就返回名称域中所有放入‘key’的变量的列表，不指定则返回所有变量。
#
# loss=tf.reduce_mean(cross_entropy)+weights_with_l2_loss
# # loss=tf.reduce_mean(cross_entropy)+weights_with_l2_loss
# #tf.reduce_mean函数用于计算张量tensor沿着指定的数轴（tensor的某一维度）上的的平均值，主要用作降维
# # 或者计算tensor（图像）的平均值。
# # reduce_mean(input_tensor,
# #             axis=None,
# #             keep_dims=False,
# #             name=None,
# #             reduction_indices=None)
# # 第一个参数input_tensor： 输入的待降维的tensor;
# # 第二个参数axis： 指定的轴，如果不指定，则计算所有元素的均值;
# # 第三个参数keep_dims：是否降维度，设置为True，输出的结果保持输入tensor的形状，设置为False，输出结果会降低维度;
# # 第四个参数name： 操作的名称;
# # 第五个参数reduction_indices：在以前版本中用来指定轴，已弃用;
# train_op=tf.train.AdamOptimizer(1e-3).minimize(loss)
# # train_op=tf.train.AdamOptimizer(0.001).minimize(loss)
# #tf.train.AdamOptimizer()函数是Adam优化算法：是一个寻找全局最优点的优化算法，引入了二次方梯度校正。
# # tf.train.AdamOptimizer.__init__(
# # 	learning_rate=0.001,
# # 	beta1=0.9,
# # 	beta2=0.999,
# # 	epsilon=1e-08,
# # 	use_locking=False,
# # 	name='Adam'
# # )
# # 参数：
# # learning_rate:张量或浮点值。学习速率
# # beta1:一个浮点值或一个常量浮点张量。一阶矩估计的指数衰减率
# # beta2:一个浮点值或一个常量浮点张量。二阶矩估计的指数衰减率
# # epsilon:数值稳定性的一个小常数
# # use_locking:如果True，要使用lock进行更新操作
# # name:应用梯度时为了创建操作的可选名称。默认为“Adam”
#
# #函数tf.nn.in_top_k()用来计算输出结果中top k的准确率，函数默认的k值是1，即top 1的准确率，也就是输出分类准确率最高时的数值
# top_k_op=tf.nn.in_top_k(result,y_,1)
# # top_k_op=tf.nn.in_top_k(result,y_,1)
# #tf.nn.in_top_k主要是用于计算预测的结果和实际结果的是否相等，返回一个bool类型的张量，
# # tf.nn.in_top_k(prediction, target, K):prediction就是表示你预测的结果，大小就是预测样本的数量乘以输出的维度，
# # 类型是tf.float32等。target就是实际样本类别的标签，大小就是样本数量的个数。
# # K表示每个样本的预测结果的前K个最大的数里面是否含有target中的值。一般都是取1。
# # top_k_op=tf.nn.in_top_k(result,y_,1)
#
# init_op=tf.global_variables_initializer()
# # init_op=tf.global_variables_initializer()
#
#
# with tf.Session() as sess:
#     sess.run(init_op)
#     #启动线程操作，这是因为之前数据增强的时候使用train.shuffle_batch()函数的时候通过参数num_threads()配置了16个线程用于组织batch的操作
#     tf.train.start_queue_runners()
#     # tf.train.start_queue_runners()
#     #QueueRunner类用来启动tensor的入队线程，可以用来启动多个工作线程同时将多个tensor（训练数据）推送入文件名
#     #称队列中，具体执行函数是 tf.train.start_queue_runners ， 只有调用 tf.train.start_queue_runners 之后，
#     #才会真正把tensor推入内存序列中，供计算单元调用，否则会由于内存序列为空，数据流图会处于一直等待状态。
#
# #每隔100step会计算并展示当前的loss、每秒钟能训练的样本数量、以及训练一个batch数据所花费的时间
#     for step in range (max_steps):      #max_steps=4000
#         start_time=time.time()  #用time来计算一下程序执行的时间:
#         image_batch,label_batch=sess.run([images_train,labels_train])
#         _,loss_value=sess.run([train_op,loss],feed_dict={x:image_batch,y_:label_batch})
#         duration=time.time() - start_time   #两次调用之间的时间差。
#
#         if step % 100 == 0:
#             examples_per_sec=batch_size / duration
#             sec_per_batch=float(duration)
#             print("step %d,loss=%.2f(%.1f examples/sec;%.3f sec/batch)"%(step,loss_value,examples_per_sec,sec_per_batch))
#
# #计算最终的正确率
#     num_batch=int(math.ceil(num_examples_for_eval/batch_size))  #math.ceil()函数用于求整
#     #num_examples_for_eval=10000,batch_size=100-----num_batch=100)
#     true_count=0
#     total_sample_count=num_batch * batch_size       #100*100?
#
#     #在一个for循环里面统计所有预测正确的样例个数
#     for j in range(num_batch):
#         image_batch,label_batch=sess.run([images_test,labels_test])
#         predictions=sess.run([top_k_op],feed_dict={x:image_batch,y_:label_batch})
#         true_count += np.sum(predictions)
#
#     #打印正确率信息
#     print("accuracy = %.3f%%"%((true_count/total_sample_count) * 100))
#
#     # num_batch=int(math.ceil(num_examples_for_eval/batch_size))
#     # true_count=0
#     # total_sample_count=num_batch*batch_size
#     # for j in range(num_batch):
#     #     image_batch,label_batch=sess.run([images_test,labels_test])
#     #     predictions=sess.run([top_k_op],feed_dict={x:image_batch,y_:label_batch})
#     #     true_count+=np.sum(predictions)
#     #
#     # print("accuracy = %.3f%%"%((true_count/total_sample_count)*100))


import tensorflow as tf
import numpy as np
import time
import math
import Cifar10_data

max_steps=4000
batch_size=100
num_ex_eval=10000
data_dir="Cifar_data/cifar-10-batches-bin"          #数据的目录

def var_with_w_loss(shape,stddev,w1):
    var=tf.Variable(tf.truncated_normal(shape,stddev=stddev))
    if w1 is not None:
        weights_loss=tf.multiply(tf.nn.l2_loss(var),w1,name="weight_loss")
        tf.add_to_collection("losses",weights_loss)
    return var
images_train,labels_train=Cifar10_data.inputs(data_dir=data_dir,batch_size=batch_size,distorted=True)
images_test,labels_test=Cifar10_data.inputs(data_dir=data_dir,batch_size=batch_size,distorted=None)

x=tf.placeholder(tf.float32,[batch_size,24,24,3])
y_=tf.placeholder(tf.int32,[batch_size])

kennel1=var_with_w_loss(shape=[5,5,3,64],stddev=52-2,w1=0.0)
conv1=tf.nn.conv2d(x,kennel1,[1,1,1,1],padding="SAME")
bias1=tf.Variable(tf.constant(0.0,shape=[64]))
relu1=tf.nn.relu(tf.nn.bias_add(conv1,bias1))
pool1=tf.nn.max_pool(relu1,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME")

kennel2=var_with_w_loss(shape=[5,5,64,64],stddev=52-2,w1=0.0)
conv2=tf.nn.conv2d(x,kennel1,[1,1,1,1],padding="SAME")
bias2=tf.Variable(tf.constant(0.1,shape=[64]))
relu2=tf.nn.relu(tf.nn.bias_add(conv1,bias1))
pool2=tf.nn.max_pool(relu1,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME")

reshape=tf.reshape(pool2,[batch_size,-1])
dim=reshape.get_shape()[1].value

weight1=var_with_w_loss(shape=[dim,384],stddev=0.04,w1=0.004)
fc_b1=tf.Variable(tf.constant(0.1,shape=[384]))
fc1=tf.nn.relu(tf.matmul(reshape,weight1)+fc_b1)


weight2=var_with_w_loss(shape=[384,192],stddev=0.04,w1=0.004)
fc_b2=tf.Variable(tf.constant(0.1,shape=[192]))
fc2=tf.nn.relu(tf.matmul(fc1,weight2)+fc_b2)

weight3=var_with_w_loss(shape=[192,10],stddev=1/192.0,w1=0.0)
fc_b3=tf.Variable(tf.constant(0.1,shape=[10]))
result=tf.nn.relu(tf.matmul(fc2,weight3)+fc_b3)

cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result,labels=(tf.cast(y_,tf.int64)))
weights_wi_l2_loss=tf.add_n(tf.get_collection("losses"))
loss=tf.reduce_mean(cross_entropy)+weights_wi_l2_loss
train_op=tf.train.AdamOptimizer(0.003).minimize(loss)  #是一个寻找全局最优点的优化算法

top_k_op=tf.nn.in_top_k(result,y_,1)
init_op=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    tf.train.start_queue_runners()
    for step in range(max_steps):
        start_time=time.time()
        image_batch,label_batch=sess.run([images_train,labels_train])
        _,loss_value=sess.run([train_op,loss],feed_dict={x:image_batch,y_:label_batch})
        dur=time.time()-start_time

        if step%100==0:
            ex_p_s=batch_size/dur
            sex_p_batch=float(dur)
            print("step %d,loss=%.2f(%.1f ex/sec;%3.f sec/batch"%(step,loss_value,ex_p_s,sex_p_batch))
            # print("step %d,loss=%.2f(%.1f examples/sec;%.3f sec/batch)"%(step,loss_value,examples_per_sec,sec_per_batch))
    # num_batch=int(math.ceil(num_examples_for_eval/batch_size))
    num_batch=int(math.ceil(num_ex_eval/batch_size))  #math.ceil()函数用于求整


    true_count=0
    total_s_count=num_batch*batch_size
    for j in range(num_batch):
        image_batch,label_batch=sess.run[images_test,labels_test]
        prediction=sess.run([top_k_op],feed_dict={x:image_batch,y_:label_batch})
        true_count+=np.sum(prediction)

    print("accuracy=%.3f%%"%((true_count/total_s_count)*100))
    # print("accuracy = %.3f%%"%((true_count/total_sample_count)*100))