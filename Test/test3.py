"""
@author:tiger
@file:test3.py
@time:2022/07/16
"""

import gym

env_name = "LunarLander-v2"
env = gym.make(env_name)          # 导入注册器中的环境

episodes = 10
for episode in range(1, episodes + 1):
    state = env.reset()           # gym风格的env开头都需要reset一下以获取起点的状态
    done = False
    score = 0

    while not done:
        env.render()              # 将当前的状态化成一个frame，再将该frame渲染到小窗口上
        action = env.action_space.sample()     # 通过随机采样获取一个随即动作
        n_state, reward, done, info = env.step(action)    # 将动作扔进环境中，从而实现和模拟器的交互
        score += reward
    print("Episode : {}, Score : {}".format(episode, score))

env.close()     # 关闭窗口