import random
import cv2
import torch
import numpy as np
from torch import nn


# 定义深度强化网络
class DQN(nn.Module):
    def __init__(self) -> None:
        super(DQN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=1600, out_features=256),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.out = nn.Linear(in_features=256, out_features=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(1, x.size(0)*x.size(1)*x.size(2))
        x = self.fc1(x)
        x = self.out(x)

        return x


# 定义Q值计算器
def Qcompute(reward, next_state, manual_reward, discount_factor=0.99):
    Q_value = reward + discount_factor * torch.max(next_state) + manual_reward

    return Q_value


# 实现经验回放机制
def replay_buffer(reply_memory, current_state, max_epoch):
    if len(reply_memory) < max_epoch:
        reply_memory.append(current_state)
    else:
        reply_memory.popleft()
        reply_memory.append(current_state)

    return reply_memory, len(reply_memory)


# 实现随机取样器
def random_sample_pool(replay_memory, buffer_size):
    index = random.randint(1, buffer_size - 1)
    random_sample = replay_memory[index-1]
    next_random_sample = replay_memory[index]

    return random_sample, next_random_sample


# 定义损失函数
DQNloss = nn.SmoothL1Loss()


# 预处理模块
def preprocess(observation):
    observation = cv2.cvtColor(cv2.resize(observation, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, observation = cv2.threshold(observation, 1, 255, cv2.THRESH_BINARY)
    return np.reshape(observation, (1, 80, 80))


# 连续图像拼接
def stackObservation(observation0, observation1, observation2, observation3):
    observation = np.stack((observation0, observation1, observation2, observation3), axis=0)
    return observation
