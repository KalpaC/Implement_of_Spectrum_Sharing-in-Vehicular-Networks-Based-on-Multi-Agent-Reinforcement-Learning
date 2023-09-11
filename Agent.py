# 作者 Ajex
# 创建时间 2023/5/6 0:47
# 文件名 Agent.py
import time

import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np

import Environment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)


class Agent:
    def __init__(self, veh_index, des_index, state_dim, n_action, memory_size):
        self.veh_index = veh_index
        self.des_index = des_index
        self.state_dim = state_dim
        # 剩余载荷1+剩余时间1+干扰功率M+信道增益4*M+迭代次数1+探索率1
        self.n_action = n_action
        # M个信道*4种功率
        self.dqn = DQN(self.state_dim, self.n_action, memory_size, batch_size=4096)
        self.transition_buffer = []

    def act(self, local_obs, epsilon):
        # 1. 将状态转化为tensor，并添加“指纹”
        # 2. 将状态输入神经网络，神经网络按照epsilon-greedy给出动作
        # 3. 返回动作
        state = torch.FloatTensor(local_obs)
        action_num = self.dqn.choose_action(state, epsilon)
        action = (action_num // 4, action_num % 4)
        self.transition_buffer.append(state)
        self.transition_buffer.append(torch.FloatTensor([action_num]))
        return action

    def remember_latest(self, global_reward, new_local_obs):
        self.transition_buffer.append(torch.FloatTensor([global_reward]))
        self.transition_buffer.append(torch.FloatTensor(new_local_obs))
        transition_vector = torch.cat(self.transition_buffer, dim=0)
        self.dqn.add_transition(transition_vector)
        self.transition_buffer = []

    def train(self):
        return self.dqn.learn_with_min_batch()

    def get_net_state(self):
        return self.dqn.eval_net.state_dict()

    def read_net_state(self, net_state_dict):
        self.dqn.eval_net.load_state_dict(net_state_dict)


class Net(nn.Module):
    def __init__(self, state_dim, n_actions):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(state_dim, 500)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(500, 250)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3 = nn.Linear(250, 120)
        self.fc3.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(120, n_actions)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        # Q-Network的输出应为给定状态下的每个状态的q值
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = func.relu(self.fc3(x))
        x = func.relu(self.out(x))
        return x


class DQN(object):
    def __init__(self, state_dim, n_action, memory_size, batch_size):
        self.eval_net, self.target_net = Net(state_dim, n_action).to(device), Net(state_dim, n_action).to(device)
        self.target_net.eval()
        self.state_dim, self.action_dim = state_dim, n_action
        self.learn_cnt = 0
        self.copy_cycles = 400
        self.memory = ReplayMemory(memory_size, self.state_dim * 2 + 2)
        self.learning_rate = 0.001
        self.optimizer = torch.optim.RMSprop(self.eval_net.parameters(), lr=self.learning_rate, momentum=0.05, eps=0.01)
        self.gamma = 1  # 需要进一步确定折扣率
        self.loss_func = nn.MSELoss()
        self.batch_size = batch_size

    def choose_action(self, state_vector, epsilon):
        # epsilon-greedy
        state_vector = state_vector.to(device)
        if np.random.uniform(0, 1) < epsilon:
            return np.random.randint(0, self.action_dim)
        else:
            with torch.no_grad():
                q_value = self.eval_net.forward(state_vector)
                action_idx = torch.argmax(q_value).item()
                return action_idx

    def add_transition(self, transition):
        self.memory.add(transition)

    def learn_with_min_batch(self):
        if self.learn_cnt % self.copy_cycles == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_cnt += 1
        transitions = self.memory.choose(self.batch_size)
        size = len(transitions)
        # memory中数据按行存储，所以取出来的经验是行优先的，所以需要按列取数据
        old_states = transitions[:, :self.state_dim]
        actions = transitions[:, self.state_dim:self.state_dim + 1].long()
        rewards = transitions[:, self.state_dim + 1:self.state_dim + 2]
        new_states = transitions[:, self.state_dim + 2:]
        q_eval = self.eval_net(old_states).gather(1, actions)
        q_next = self.target_net(new_states).detach()
        q_target = rewards + self.gamma * q_next.max(1)[0].view(size, 1)
        loss = self.loss_func(q_eval, q_target)
        # 计算new_states的q值
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


class ReplayMemory:
    def __init__(self, size, transition_dim):
        self.size, self.transition_dim = size, transition_dim
        self.counter = 0
        self.mem = torch.zeros(size, transition_dim).to(device)

    def add(self, transition):
        self.mem[self.counter % self.size] = transition
        self.counter = self.counter + 1

    def choose(self, batch_size):
        if self.counter < batch_size:
            sample_index = range(0, self.counter)
        else:
            sample_index = np.random.choice(min(self.counter, self.size), batch_size)
        return self.mem[sample_index]
