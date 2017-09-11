
# coding: utf-8
import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Xavier初始化器，用来让权重被初始化的不大不小，正好合适
# Xavier就是让权重满足0均值，同时方差为2/(n_in+n_out)，分布可以是均匀分布或高斯分布
# fan_in：输入节点的数量
# fan_out:输出节点的数量
def xavier_init(fan_in,fan_out,constant=1):
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in,fan_out),
                            minval = low,maxval = high,
                            dtype = tf.float32)

# 去噪自编码的类：包含一个构造函数和几个成员函数
class AdditiveGaussianNoiseAutoEncoder(object):
    # 构造函数，用于初始化
    # n_input:输入变量数
    # n_hidden:隐含层节点数
    # transfer_function:隐含层激活函数，默认为softplus
    # optimizer:优化器，默认为Adam
    # scale:高斯噪声系数，默认为0.1
    # 其中，scale参数做成了一个placeholder，参数初始化使用了接下来定义的_initialize_weights()函数
    def __init__(self,n_input,n_hidden,transfer_function = tf.nn.softplus,
                optimizer = tf.train.AdamOptimizer(),scale = 0.1):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        network_weights = self._initialize_weights()
        self.weights = network_weights
        
        # 定义网络结构
        '''
            首先为输入x创建一个维度为n_input的placeholder，然后建立一个能提取特征的隐含层。
            我们先将输入x加入噪声，即self.x + scale*tf.random_normal((n_input,)),
            然后用tf.matmul将加了噪声的输入与隐含层的权重w1相乘，并使用tf.add加上隐含层的偏置b1，
            最后使用self.transfer对结果进行激活函数处理；
            经过隐含层后，需要在输出层进行数据复原、重建操作（即建立reconstruction层）；
            这里不需要激活函数了，直接将隐含层的输出self.hidden乘上输出层的权重w2,再加上输出层的偏置b2即可。
        '''
        self.x = tf.placeholder(tf.float32,[None,self.n_input])
        self.hidden = self.transfer(tf.add(tf.matmul(
                self.x + scale*tf.random_normal((n_input,)),
                self.weights['w1']),self.weights['b1']))
        self.reconstruction = tf.add(tf.matmul(self.hidden,self.weights['w2']),self.weights['b2'])
        
        # 定义自编码器的损失函数
        '''
          这里直接使用平方误差作为cost,即用tf.subtract计算输出（self.reconstruction）与输入（self.x）之差,
          再使用tf.pow来求差的平方，最后使用tf.reduce_sum求和即可得到平方误差；
          再定义训练操作为优化器self.optimizer对损失self.cost进行优化；
          最后建立session,并初始化自编码器的全部模型参数。
        '''
        # 损失函数
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction,self.x),2.0))
        self.optimizer = optimizer.minimize(self.cost)
        
        # 进行全部模型参数的初始化
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        
    # 参数初始化函数
    '''
      先创建一个名为all_weights的字典dict,然后将w1,b1,w2,b2全部存入其中，最后返回all_weights。
      其中，w1需要使用前面定义的xavier_init函数初始化，
      直接传入输入节点数和隐含层节点数，然后xavier即可返回一个比较适合于softplus等激活函数的权重初始分布，
      而偏置b1只需要使用tf.zeros全部置为0即可；
      对于输出层self.reconstruction，因为没有使用激活函数，这里将w2,b2全部初始化为0即可。
    '''
    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input,self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden],dtype = tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden,self.n_input],dtype = tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input],dtype = tf.float32))
        
        return all_weights
    
    # 定义计算损失cost及执行一步训练的函数partial_fit;
    # 函数里只需让Session执行两个计算图的节点，分别是损失cost和训练过程optimizer，
    # 输入的feed_dict包括输入数x,以及噪声的系数scale。
    def partial_fit(self,X):
        cost,opt = self.sess.run((self.cost,self.optimizer),
                                feed_dict = {self.x:X,self.scale:self.training_scale})
        #返回当前的损失
        return cost
    
    # 定义一个只求损失cost的函数
    '''
        只让Session执行一个计算图节点self.cost,
        传入的参数和partial_fit一致；
        在自编码器训练完毕后，在测试集上对模型性能进行评测时会用到，但是不会像partial_fit那样触发训练操作
    '''
    def calc_total_cost(self,X):
        return self.sess.run(self.cost,feed_dict = {self.x:X,self.scale:self.training_scale})
    
    # 返回隐含层的输出结果
    '''
        目的是提供一个接口来获取抽象后的特征，
        自编码器的隐含层的最主要功能就是学习出数据中的高阶特征
    '''
    def transfer(self,X):
        return self.sess.run(self.hidden,feed_dict = {self.x:X,self.scale:self.training_scale})
    
    # 将隐含层的输出结果作为输入，通过之后的重建层将提取到的高阶特征复原为原始特征
    '''
       这个接口和前面的transfer正好将整个自编码器拆分为两部分，
       这里的generate是后半部分，将高阶的特征复原为原始数据
    '''
    def generate(self,hidden = None):
        if hidden is None:
            hidden = np.random.normal(size = self.weights["b1"])
        return self.sess.run(self.reconstruction,feed_dict = {self.hidden:hidden})
    
    # 定义reconstruct函数，它整体运行一遍复原过程，包括提取高阶特征和通过高阶特征来复原数据
    # 即包括transfer和generate两块，输入数据是原始数据，输出数据是复原后的数据
    def reconstruct(self,X):
        return self.sess.run(self.reconstruction,feed_dict = {self.x:X,
                                                             self.scale:training_scale})
    
    # 获取隐含层的权重w1
    def getWeights(self):
        return self.sess.run(self.weights['w1'])
    
    # 获取隐含层的偏置系数b1
    def getBiases(self):
        return self.sess.run(self.weights['b1'])    



# 测试
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

# 定义一个对训练、测试数据进行标准化处理的函数
'''
    标准化即让数据变成0均值，且标准差为1的分布。
    方法就是减去均值，再除以标准差：直接使用sklearn.preprossing的StandarScaler这个类，
    先在训练集上进行fit,再将这个Scaler用到训练数据和测试数据上。
'''
def standard_scale(X_train,X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    
    return X_train,X_test

# 定义一个获取随机block数据的函数
'''
   取一个从0到len(data) - batch_size之间的随机整数，
   再以这个随机数作为block的起始位置，然后顺序取到一个batch_size的数据
'''
def get_random_block_from_data(data,batch_size):
    start_index = np.random.randint(0,len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]

# 使用之前定义的standard_scale函数来对训练、测试数据进行标准化变换
X_train,X_test = standard_scale(mnist.train.images,mnist.test.images)

# 定义几个参数
n_samples = int(mnist.train.num_examples) # 总训练样本数
training_epochs = 100 # 最大训练轮数
batch_size = 128 
display_step = 1 # 每隔一轮显示一次损失

# 创建AGN的实例
'''
   定义模型的输入节点数n_input为784，隐含层节点数为200，隐含层的激活函数为softplus,
   优化器为Adam且学习速率为0.001，同时噪声的系数scale设为0.01
'''
autoencoder = AdditiveGaussianNoiseAutoEncoder(n_input = 784,
                                              n_hidden = 200,
                                              transfer_function = tf.nn.softplus,
                                              optimizer = tf.train.AdamOptimizer(learning_rate = 0.001),
                                              scale = 0.01)

# 开始训练
'''
    在每一轮循环的开始时，先将平均损失avg_cost设为0，并计算总共需要的batch数；
    然后在每一个batch的循环中，先使用get_random_block_from_data函数随机抽取一个block的数据，
    然后使用成员函数partial_fit训练这个batch的数据并计算当前的cost，最后将当前的cost整合到avg_cost中。
    在每一轮迭代的最后，显示当前的迭代数和这一轮迭代的平均cost.
'''
for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(n_samples / batch_size)
    for i in range(total_batch):
        batch_xs = get_random_block_from_data(X_train,batch_size)
        
        cost = autoencoder.partial_fit(batch_xs)
        avg_cost += cost/n_samples*batch_size
    
    if epoch % display_step == 0:
        print("Epoch:",'%04d'%(epoch+1),"cost=","{:.9f}".format(avg_cost))

# 对训练完的模型进行性能测试，使用之前定义的cal_total_cost函数对测试集X_test进行测试，评价指标是平方误差
print("Total cost: " + str(autoencoder.calc_total_cost(X_test)))