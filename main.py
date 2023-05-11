from model.DQN_NET import *
import torch
import torch.optim as optim
import numpy as np
import game.wrapped_flappy_bird as game
from collections import deque
 
# 设定超参数
MAX_TRAIN_EPOCH = 1000000
LEARNING_RATE = 0.0005
EPSILON = 1
DISCOUNT_FACTOR = 0.99
BATCH_SIZE = 128
BUFFER_SIZE = 10000
BUFFER_BATCH = 100
MANUAL_REWARD_WEIGHT = 0.1
LR_STEP_SIZE = 100
LR_GAMMA = 0.5
PARAMS_UPDATE_SIZE = 100

# 开始训练
if __name__ == '__main__':
    # 创建主网络和目标网络
    flappyBird_DQN = DQN()
    target_flappyBird_DQN = DQN()
    DQN.load_state_dict(self=DQN(), state_dict=torch.load('weight/weight33000.pth'))

    # 拷贝主网络参数至目标网络
    target_flappyBird_DQN.load_state_dict(flappyBird_DQN.state_dict())

    optimizer = optim.SGD(flappyBird_DQN.parameters(), lr=LEARNING_RATE, weight_decay=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

    # 创建经验池
    replay_memory = deque(maxlen=MAX_TRAIN_EPOCH)

    for epoch in range(MAX_TRAIN_EPOCH):
        epoch = epoch + 33000
        print(f'TRAIN_EPOCH : {epoch}')

        # 初始化第一次行为
        flag = False
        action0 = np.array([1, 0])
        flappyBird = game.GameState()
        observation, reward0, terminal = flappyBird.frame_step(action0)
        observation = preprocess(observation)
        observation = stackObservation(observation, observation, observation, observation)
        observation = torch.from_numpy(observation).reshape((4, 80, 80)).to(torch.float)

        if epoch % PARAMS_UPDATE_SIZE == 0:
            EPSILON = EPSILON * 0.5

        while not flag:
            # 随机探索机制
            if np.random.rand() > EPSILON:
                q_chart = flappyBird_DQN(observation)
                q_value = torch.max(q_chart)
                action = torch.where(q_chart == q_value, 1, 0).numpy()
                action = np.reshape(action, (2, ))
            else:
                q_chart = flappyBird_DQN(observation)
                q_value = torch.min(q_chart)
                action = torch.where(q_chart == q_value, 1, 0).numpy()
                action = np.reshape(action, (2, ))

            # 计算当前状态的四帧图像
            observation0, reward0, terminal = flappyBird.frame_step(action)
            if terminal:
                flag = terminal
            observation1, reward1, terminal = flappyBird.frame_step(action0)
            if terminal:
                flag = terminal
            observation2, reward2, terminal = flappyBird.frame_step(action0)
            if terminal:
                flag = terminal
            observation3, reward3, terminal = flappyBird.frame_step(action0)
            if terminal:
                flag = terminal

            # 计算目标网络q值
            observation0, observation1, observation2, observation3 = preprocess(observation0), preprocess(observation1), preprocess(observation2), preprocess(observation3)
            observation = stackObservation(observation0, observation1, observation2, observation3)
            observation = torch.from_numpy(observation).reshape((4, 80, 80)).to(torch.float)

            # 更新目标网络
            if epoch % PARAMS_UPDATE_SIZE == 0:
                target_flappyBird_DQN.load_state_dict(flappyBird_DQN.state_dict())

            with torch.no_grad():
                next_q_value = target_flappyBird_DQN(observation)

            target_q_value = Qcompute(q_value, next_q_value, MANUAL_REWARD_WEIGHT * reward3, DISCOUNT_FACTOR)

            # 损失函数
            loss = DQNloss(q_value, target_q_value)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 将当前状态存入经验池
            current_state = observation, reward3, terminal
            replay_memory, buffer_size = replay_buffer(replay_memory, current_state, BATCH_SIZE)

            # 经验回放
        for count in range(BUFFER_BATCH):
            (observation, reward, replay_terminal), (next_observation, next_reward, netx_reply_terminal) = random_sample_pool(replay_memory, buffer_size)

            q_chart = flappyBird_DQN(observation)
            q_value = torch.max(q_chart)

            with torch.no_grad():
                next_q_value = flappyBird_DQN(next_observation)

            target_q_value = Qcompute(q_value, next_q_value, MANUAL_REWARD_WEIGHT * next_reward, DISCOUNT_FACTOR)

            # 损失函数
            loss = DQNloss(q_value, target_q_value)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 保存参数
        if epoch % 100 == 0:
            print(f'loss: {loss.item()}')
        if epoch > 5000 and epoch % 1000 == 0:
            torch.save(flappyBird_DQN.state_dict(), f'weight/weight{epoch}.pth')
            print('save successfully')









