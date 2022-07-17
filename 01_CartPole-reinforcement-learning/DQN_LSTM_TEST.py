"""
@author:tiger
@file:DQN_LSTM_TEST.py
@time:2022/07/17
"""

import parl
from parl.utils import logger
import paddle
import copy
import numpy as np
import os
import gym
import random
import collections

#设置会用到的超参数
learn_freq = 3 # 训练频率，不需要每一个step都learn，攒一些新增经验后再learn，提高效率
memory_warmup_size = 50  # episode_replay_memory 里需要预存一些经验数据，再开启训练
batch_size = 8   # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
lr = 6e-4 # 学习率
gamma = 0.99 # reward 的衰减因子，一般取 0.9 到 0.999 不等
num_step=10
episode_size=500    # replay memory的大小(数据集的大小)，越大越占用内存

#搭建网络
class Model(paddle.nn.Layer):
    def __init__(self, obs_dim,act_dim):
        super(Model,self).__init__()
        self.hidden_size=64
        self.first=False
        self.act_dim=act_dim
        # 3层全连接网络
        self.fc1 =  paddle.nn.Sequential(
                                        paddle.nn.Linear(obs_dim,128),
                                        paddle.nn.ReLU())

        self.fc2 = paddle.nn.Sequential(
                                        paddle.nn.Linear(self.hidden_size,128),
                                        paddle.nn.ReLU())
        self.fc3 = paddle.nn.Linear(128,act_dim)
        self.lstm=paddle.nn.LSTM(128,self.hidden_size,1)      #[input_size,hidden_size,num_layers]

    def init_lstm_state(self,batch_size):
        self.h=paddle.zeros(shape=[1,batch_size,self.hidden_size],dtype='float32')
        self.c=paddle.zeros(shape=[1,batch_size,self.hidden_size],dtype='float32')
        self.first=True

    def forward(self, obs):
        # 输入state，输出所有action对应的Q，[Q(s,a1), Q(s,a2), Q(s,a3)...]
        obs = self.fc1(obs)
        #每次训练开始前重置
        if (self.first):
            x,(h,c) = self.lstm(obs,(self.h,self.c))  #obs:[batch_size,num_steps,input_size]
            self.first=False
        else:
            x,(h,c) = self.lstm(obs)  #obs:[batch_size,num_steps,input_size]
        x=paddle.reshape(x,shape=[-1,self.hidden_size])
        h2 = self.fc2(x)
        Q = self.fc3(h2)
        return Q

#DRQN算法
class DRQN(parl.Algorithm):
    def __init__(self, model, act_dim=None, gamma=None, lr=None):
        self.model = model
        self.target_model = copy.deepcopy(model)    #复制predict网络得到target网络，实现fixed-Q-target 功能

        #数据类型是否正确
        assert isinstance(act_dim, int)
        assert isinstance(gamma, float)
        assert isinstance(lr, float)

        self.act_dim = act_dim
        self.gamma = gamma
        self.lr = lr
        self.optimizer=paddle.optimizer.Adam(learning_rate=self.lr,parameters=self.model.parameters())    # 使用Adam优化器

    #预测功能
    def predict(self, obs):
        return self.model.forward(obs)

    def learn(self, obs, action, reward, next_obs, terminal):
        #将数据拉平
        action=paddle.reshape(action,shape=[-1])
        reward=paddle.reshape(reward,shape=[-1])
        terminal=paddle.reshape(terminal,shape=[-1])

        # 从target_model中获取 max Q' 的值，用于计算target_Q
        next_predict_Q = self.target_model.forward(next_obs)
        best_v = paddle.max(next_predict_Q, axis=-1)#next_predict_Q的每一个维度（行）都求最大值，因为每一行就对应一个St,行数就是我们输入数据的批次大小
        best_v.stop_gradient = True                 #阻止梯度传递,因为要固定模型参数
        terminal = paddle.cast(terminal, dtype='float32')    #转换数据类型，转换为float32
        target = reward + (1.0 - terminal) * self.gamma * best_v  #Q的现实值

        predict_Q = self.model.forward(obs)  # 获取Q预测值

        #接下来一步是获取action所对应的Q(s,a)
        action_onehot = paddle.nn.functional.one_hot(action, self.act_dim)    # 将action转onehot向量，比如：3 => [0,0,0,1,0]
        action_onehot = paddle.cast(action_onehot, dtype='float32')
        predict_action_Q = paddle.sum(
                                      paddle.multiply(action_onehot, predict_Q)              #逐元素相乘，拿到action对应的 Q(s,a)
                                      , axis=1)  #对每行进行求和运算,注意此处进行求和的真正目的其  # 比如：pred_value = [[2.3, 5.7, 1.2, 3.9, 1.4]], action_onehot = [[0,0,0,1,0]]
                                                #实是变换维度，类似于矩阵转置。与target形式相同。 #  ==> pred_action_value = [[3.9]]


        # 计算 Q(s,a) 与 target_Q的均方差，得到损失。让一组的输出逼近另一组的输出，是回归问题，故用均方差损失函数

        loss=paddle.nn.functional.square_error_cost(predict_action_Q, target)
        cost = paddle.mean(loss)
        cost.backward()   #反向传播
        self.optimizer.step()  #更新参数
        self.optimizer.clear_grad()  #清除梯度

    def sync_target(self):
        self.target_model = copy.deepcopy(model)    #复制predict网络得到target网络，实现fixed-Q-target 功能


class Agent(parl.Agent):
    def __init__(self,
                 algorithm,
                 act_dim,
                 e_greed=0.1,
                 e_greed_decrement=0 ):

        #判断输入数据的类型是否是int型
        assert isinstance(act_dim, int)

        self.act_dim = act_dim

        #调用Agent父类的对象，将算法类algorithm输入进去,目的是我们可以调用algorithm中的成员
        super(Agent, self).__init__(algorithm)

        self.global_step = 0          #总运行步骤
        self.update_target_steps = 200  # 每隔200个training steps再把model的参数复制到target_model中

        self.e_greed = e_greed  # 有一定概率随机选取动作，探索
        self.e_greed_decrement = e_greed_decrement  # 随着训练逐步收敛，探索的程度慢慢降低

    #参数obs都是单条输入,与learn函数的参数不同
    def sample(self, obs):
        sample = np.random.rand()  # 产生0~1之间的小数
        if sample < self.e_greed:
            act = np.random.randint(self.act_dim)  # 探索：每个动作都有概率被选择
        else:
            act = self.predict(obs)  # 选择最优动作
        self.e_greed = max(
            0.01, self.e_greed - self.e_greed_decrement)  # 随着训练逐步收敛，探索的程度慢慢降低
        return act

    #通过神经网络获取输出
    def predict(self, obs):  # 选择最优动作
        obs=paddle.to_tensor(obs,dtype='float32')  #将目标数组转换为张量
        predict_Q=self.alg.predict(obs).numpy()    #将结果张量转换为数组
        act = np.argmax(predict_Q)  # 选择Q最大的下标，即对应的动作
        return act

    #这里的learn函数主要包括两个功能。1.同步模型参数2.更新模型。这两个功能都是通过调用algorithm算法里面的函数最终实现的。
    #注意，此处输入的参数均是一批数据组成的数组
    def learn(self, obs, act, reward, next_obs, terminal):
        # 每隔200个training steps同步一次model和target_model的参数
        if self.global_step % self.update_target_steps == 0:
            self.alg.sync_target()
        self.global_step += 1      #每执行一次learn函数，总次数+1

        #转换为张量
        obs=paddle.to_tensor(obs,dtype='float32')
        act=paddle.to_tensor(act,dtype='int32')
        reward=paddle.to_tensor(reward,dtype='float32')
        next_obs=paddle.to_tensor(next_obs,dtype='float32')
        terminal=paddle.to_tensor(terminal,dtype='float32')

        #进行学习
        self.alg.learn(obs, act, reward, next_obs, terminal)


class EpisodeMemory(object):
    def __init__(self,episode_size,num_step):
        self.buffer = collections.deque(maxlen=episode_size)
        self.num_step=num_step   #时间步长

    def put(self,episode):
        self.buffer.append(episode)

    def sample(self,batch_size):
        mini_batch = random.sample(self.buffer, batch_size)  #返回值是个列表
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = [], [], [], [], []

        for experience in mini_batch:
            self.num_step = min(self.num_step, len(experience)) #防止序列长度小于预定义长度

        for experience in mini_batch:
            idx = np.random.randint(0, len(experience)-self.num_step+1)  #随机选取一个时间步的id
            s, a, r, s_p, done = [],[],[],[],[]
            for i in range(idx,idx+self.num_step):
                e1,e2,e3,e4,e5=experience[i][0]
                s.append(e1[0][0]),a.append(e2),r.append(e3),s_p.append(e4),done.append(e5)
            obs_batch.append(s)
            action_batch.append(a)
            reward_batch.append(r)
            next_obs_batch.append(s_p)
            done_batch.append(done)

        #转换数据格式
        obs_batch=np.array(obs_batch).astype('float32')
        action_batch=np.array(action_batch).astype('float32')
        reward_batch=np.array(reward_batch).astype('float32')
        next_obs_batch=np.array(next_obs_batch).astype('float32')
        done_batch=np.array(done_batch).astype('float32')

        #将列表转换为数组并转换数据类型
        return obs_batch,action_batch,reward_batch,next_obs_batch,done_batch

    #输出队列的长度
    def __len__(self):
        return len(self.buffer)

class ReplayMemory(object):
    def __init__(self,e_rpm):
        #创建一个固定长度的队列作为缓冲区域，当队列满时，会自动删除最老的一条信息
        self.e_rpm=e_rpm
        self.buff=[]
    # 增加一条经验到经验池中
    def append(self,exp,done):
        self.buff.append([exp])
        #将一整个episode添加进经验池
        if(done):
            self.e_rpm.put(self.buff)
            self.buff=[]
    #输出队列的长度
    def __len__(self):
        return len(self.buff)


# 训练一个episode
def run_episode(env, agent, rpm, e_rpm, obs_shape):   #rpm就是经验池
    for step in range(1,learn_freq+1):
        #重置环境
        obs = env.reset()
        while True:
            obs=obs.reshape(1,1,obs_shape)
            action = agent.sample(obs)  # 采样动作，所有动作都有概率被尝试到
            next_obs, reward, done, _ = env.step(action)
            rpm.append((obs, action, reward, next_obs, done),done)   #搜集数据
            obs = next_obs
            if done:
                break

    #存储足够多的经验之后按照间隔进行训练
    if (len(e_rpm) > memory_warmup_size):
        #每次训练之前重置LSTM参数
        model.init_lstm_state(batch_size)
        (batch_obs, batch_action, batch_reward, batch_next_obs,batch_done) = e_rpm.sample(batch_size)
        agent.learn(batch_obs, batch_action, batch_reward,batch_next_obs,batch_done)  # s,a,r,s',done

# 评估 agent, 跑 5 个episode，总reward求平均
def evaluate(env, agent, obs_shape,render=False):
    eval_reward = []   #列表存储所有episode的reward
    for i in range(5):
        obs = env.reset()
        episode_reward = 0
        while True:
            obs=obs.reshape(1,1,obs_shape)
            action = agent.predict(obs)  # 预测动作，只选最优动作
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            if render:
                env.render()
            if done:
                break
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)  #求平均值


if __name__ == '__main__':

    env = gym.make('CartPole-v1')
    action_dim = env.action_space.n
    obs_shape = env.observation_space.shape

    save_path = './dqn_model.ckpt'

    e_rpm=EpisodeMemory(episode_size,num_step)
    rpm = ReplayMemory(e_rpm)  # 实例化DQN的经验回放池
    # 根据parl框架构建agent
    model = Model(obs_dim=obs_shape[0],act_dim=action_dim)
    algorithm = DRQN(model, act_dim=action_dim, gamma=gamma, lr=lr)
    agent = Agent(
        algorithm,
        act_dim=action_dim,
        e_greed=0.1,  # 有一定概率随机选取动作，探索
        e_greed_decrement=8e-7)  # 随着训练逐步收敛，探索的程度慢慢降低

    # 先往经验池里存一些数据，避免最开始训练的时候样本丰富度不够
    while len(e_rpm) < memory_warmup_size:
        run_episode(env, agent, rpm,e_rpm,obs_shape[0])

    #定义训练次数
    max_train_num = 2000
    best_acc=500

    try:
        agent.restore(save_path)
    except Exception as e:
        print("agent load model data error ...... ")

    # 开始训练
    train_num = 0
    while train_num < max_train_num:  # 训练max_episode个回合，test部分不计算入episode数量
        # train part
        #for循环的目的是每50次进行一下测试
        for i in range(0, 50):
            run_episode(env, agent,rpm, e_rpm,obs_shape[0])
            train_num += 1
        # test part
        eval_reward = evaluate(env, agent,obs_shape[0], render=False)  #render=True 查看显示效果

        if eval_reward>best_acc:
            best_acc=eval_reward
            agent.save(save_path)

        #将信息写入日志文件
        logger.info('train_num:{}    e_greed:{}   test_reward:{}'.format(
            train_num, agent.e_greed, eval_reward))