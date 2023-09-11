# main 2023/5/18 23:16
import time

import scipy.io
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import Environment
import random
import os
import logging
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
V2V_power_dB_List = [23, 10, 5, -100]
n_veh = 4
n_des = 1
n_RB = 4
time_slow = 0.1
time_fast = 0.001
n_episode = 3000

def get_local_state(env: Environment.Environment, idx=(0, 0), episode=1., epsilon=0.02):
    i, j = idx
    demand = env.remain_payload[i][j] / env.payload_size
    remain_time = env.remain_time / env.time_slow
    # M维
    V2V_interference_dB = env.V2V_Interference_dB[i][j]
    # print("V2V_interference_dB",V2V_interference_dB)
    # M维
    G_k = - env.V2V_channels_with_fastfading[i][env.vehicles[i].destinations[j]] \
          + env.vehAntGain * 2 \
          - env.vehNoiseFigure
    # print("G_k",G_k)
    G_other_k2k = np.zeros((env.n_veh * env.n_des - 1, env.n_RB))
    num = 0
    for k in range(env.n_veh):
        for l in range(env.n_des):
            if k == i and l == j:
                continue
            G_other_k2k[num] = -env.V2V_channels_with_fastfading[k][env.vehicles[i].destinations[j]] \
                               + env.vehAntGain * 2 \
                               - env.vehNoiseFigure
            num += 1
    # print("G_other_k2k",G_other_k2k)
    G_k_B = - env.V2I_channels_with_fastfading[i] + env.vehAntGain + env.bsAntGain - env.bsNoiseFigure
    # print("G_k_B",G_k_B)
    G_m_k = np.zeros(env.n_RB)
    for m in range(env.n_RB):
        G_m_k[m] = - env.V2V_channels_with_fastfading[m][env.vehicles[i].destinations[j]][m] \
                   + env.vehAntGain * 2 \
                   - env.vehNoiseFigure
    # print("G_m_k",G_m_k)
    # time.sleep(1)
    return np.concatenate((V2V_interference_dB,
                           G_k,  # 当前信道增益
                           G_other_k2k.flatten(),  # 其他信道的干扰
                           G_k_B,  # 当前信道对BS的干扰
                           G_m_k,  # 占据每个子载波的V2I信道对信道m在V2V接收方的干扰
                           np.asarray([demand, remain_time, episode, epsilon])))


class DQN(nn.Module):
    def __init__(self, input_size, n_hidden1, n_hidden2, n_hidden3, output_size):
        super(DQN, self).__init__()
        self.fc_1 = nn.Linear(input_size, n_hidden1)
        self.fc_1.weight.data.normal_(0, 0.1)
        self.fc_2 = nn.Linear(n_hidden1, n_hidden2)
        self.fc_2.weight.data.normal_(0, 0.1)
        self.fc_3 = nn.Linear(n_hidden2, n_hidden3)
        self.fc_3.weight.data.normal_(0, 0.1)
        self.fc_4 = nn.Linear(n_hidden3, output_size)
        self.fc_4.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = F.relu(self.fc_3(x))
        x = F.relu(self.fc_4(x))
        return x


class ReplayMemory:
    def __init__(self, entry_size):
        self.entry_size = entry_size
        self.memory_size = 400000
        self.actions = np.empty(self.memory_size, dtype=np.uint8)
        self.rewards = np.empty(self.memory_size, dtype=np.float64)
        self.prestate = np.empty((self.memory_size, self.entry_size), dtype=np.float16)
        self.poststate = np.empty((self.memory_size, self.entry_size), dtype=np.float16)
        self.batch_size = 4000
        self.count = 0
        self.current = 0

    def add(self, prestate, poststate, reward, action):
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.prestate[self.current] = prestate
        self.poststate[self.current] = poststate
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.memory_size

    def sample(self):
        if self.count < self.batch_size:
            indexes = range(0, self.count)
        else:
            indexes = random.sample(range(0, self.count), self.batch_size)
        prestate = self.prestate[indexes]
        poststate = self.poststate[indexes]
        actions = self.actions[indexes]
        rewards = self.rewards[indexes]
        return prestate, poststate, actions, rewards


class Agent:
    def __init__(self, input_dim, output_dim):
        self.discount = 1
        self.double_q = False
        self.memory = ReplayMemory(input_dim)
        self.model = DQN(input_dim, 500, 250, 120, output_dim).to(device)
        self.target_model = DQN(input_dim, 500, 250, 120, output_dim).to(
            device)  # Target Model
        self.target_model.eval()
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=0.001, momentum=0.05, eps=0.01)
        self.loss_func = nn.MSELoss()

    def predict(self, s_t, ep=0.):
        n_power_levels = len(V2V_power_dB_List)
        # state_t = torch.from_numpy(s_t).type(torch.float32).view([1, self.memory_entry_size])
        if random.random() > ep:
            with torch.no_grad():
                q_values = self.model(torch.tensor(s_t, dtype=torch.float32).unsqueeze(0).to(device))
                return q_values.max(1)[1].item()
        else:
            return random.choice(range(n_power_levels * n_RB))

    def Q_Learning_mini_batch(self):  # Double Q-Learning
        batch_s_t, batch_s_t_plus_1, batch_action, batch_reward = self.memory.sample()
        action = torch.LongTensor(batch_action).to(device)
        reward = torch.FloatTensor(batch_reward).to(device)
        state = torch.FloatTensor(np.float32(batch_s_t)).to(device)
        next_state = torch.FloatTensor(np.float32(batch_s_t_plus_1)).to(device)
        if self.double_q:
            next_action = self.model(next_state).max(1)[1]
            next_q_values = self.target_model(next_state)
            next_q_value = next_q_values.gather(1, next_action.unsqueeze(1)).squeeze(1)
            expected_q_value = reward + self.discount * next_q_value
        else:
            next_q_value = self.target_model(next_state).max(1)[0]
            expected_q_value = reward + self.discount * next_q_value
        q_values = self.model(state)
        q_acted = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        loss = self.loss_func(expected_q_value.detach(), q_acted)
        # backward and optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())


def get_epsilon(episode):
    if episode < 2400:
        return 1 - 0.98 / 2399 * episode
    else:
        return 0.02


def compute_reward(env, V2I_capacity, V2V_capacity):
    # 为避免奖励值过大，单位换为MBps
    reward = np.sum(V2I_capacity) / (8 * 1000 * 1000) * 0.1
    reward += sum([V2V_capacity[i][j] / (8 * 1000 * 1000) if env.remain_payload[i][j] > 0 else 10
                   for i in range(env.n_veh)
                   for j in range(env.n_des)]) * 0.9
    return reward


class Algorithm:
    def __init__(self):
        self.default_B = 2 * 1060
        pass

    def train_model(self, model_dir):
        env = Environment.Environment(n_veh, n_des, self.default_B)
        env.new_random_game()
        steps_per_episode = int(time_slow / time_fast)
        target_update_episodes = 4
        state_dim = len(get_local_state(env))
        action_dim = n_RB * len(env.V2V_power_dB_list)
        agents = []
        for i in range(n_veh * n_des):
            agents.append(Agent(state_dim, action_dim))

        # 需要可视化的内容包括：每轮loss、每轮奖励之和、每轮传输成功率、V2I总容量随episode的变化
        interval = 10
        size = int(n_episode / interval)
        loss_list = np.zeros(size, dtype='float')
        reward_list = np.zeros(size, dtype='float')
        probability_list = np.zeros(size, dtype='float')
        V2I_capacity_list = np.zeros(size, dtype='float')

        for e in range(n_episode):
            epsilon = get_epsilon(e)
            env.renew_environment()
            env.reset_payload(self.default_B)
            env.reset_time(time_slow)
            env.is_active = np.ones((n_veh, n_des), dtype='bool')
            env.compute_V2V_interference(np.zeros((n_veh, n_des, 2), dtype='int32'))
            for t in range(steps_per_episode):
                old_states = []
                actions = np.zeros((n_veh, n_des, 2), dtype='int32')
                for i in range(n_veh):
                    for j in range(n_des):
                        # 获取当前状态
                        idx = i * n_des + j
                        state = get_local_state(env, (i, j), e / (n_episode - 1), epsilon)
                        old_states.append(state)
                        epsilon_greedy_action = agents[idx].predict(state, epsilon)
                        actions[i, j, 0] = epsilon_greedy_action % n_RB
                        actions[i, j, 1] = int(epsilon_greedy_action / n_RB)
                actions__ = actions.copy()
                r_t, V2I_rate = env.act_in_env(actions__)
                V2I_capacity_list[e // interval] += V2I_rate.sum() * env.bandwidth
                reward_list[e // interval] += r_t
                env.renew_channels_fastfading()
                env.compute_V2V_interference(actions__)
                for i in range(n_veh):
                    for j in range(n_des):
                        idx = i * n_des + j
                        s_t = old_states[idx]
                        a_t = actions[i, j][0] + actions[i, j][1] * n_RB
                        s_tp = get_local_state(env, (i, j), e / (n_episode - 1), epsilon)
                        agents[idx].memory.add(s_t, s_tp, r_t, a_t)
            probability_list[e // 10] += np.count_nonzero(env.remain_payload <= 0) / (n_veh * n_des)
            for i in range(n_veh):
                for j in range(n_des):
                    idx = i * n_des + j
                    loss_batch = agents[idx].Q_Learning_mini_batch()
                    if i == 0 and j == 0:
                        loss_list[e // interval] += loss_batch
                    if e % 100 == 0 and i == 0 and j == 0:
                        print('Episode:', e, 'agent0_loss', loss_batch)
                    if e % target_update_episodes == target_update_episodes - 1:
                        agents[idx].update_target_network()
            if e % interval == 9:
                loss_list[e // interval] /= interval
                reward_list[e // interval] /= interval
                V2I_capacity_list[e // interval] /= interval * steps_per_episode
                probability_list[e // interval] /= interval
                if e % 100 == 99:
                    print('Average transmission probability', probability_list[e // interval])

        episodes = range(0, n_episode, interval)
        fig = plt.figure()
        ax1 = fig.add_subplot(221)
        ax1.set(title='MARL-loss', ylabel='loss', xlabel='episodes')
        ax1.plot(episodes, loss_list)

        ax2 = fig.add_subplot(222)
        ax2.set(title='MARL-reward', ylabel='reward', xlabel='episodes')
        ax2.plot(episodes, reward_list)

        ax3 = fig.add_subplot(223)
        ax3.set(title='MARL-V2I', ylabel='V2I_rate(MBps)', xlabel='episodes')
        ax3.plot(episodes, V2I_capacity_list)

        ax4 = fig.add_subplot(224)
        ax4.set(title='MARL-V2V', ylabel='V2V payload transmission probability', xlabel='episodes')
        ax4.plot(episodes, probability_list)
        plt.show()

        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        for i in range(n_veh):
            for j in range(n_des):
                idx = i * n_des + j
                agent: Agent = agents[idx]
                path = os.path.join(model_dir, 'agent-%d-%d.m' % (i, j))
                torch.save(agent.model.state_dict(), path)

    def test_model(self, model_dir, payload_size):
        env = Environment.Environment(n_veh, n_des, payload_size)
        env.new_random_game()
        test_episode = 100
        final_epsilon = get_epsilon(n_episode - 1)
        steps_per_episode = int(time_slow / time_fast)
        state_dim = len(get_local_state(env))
        action_dim = n_RB * len(env.V2V_power_dB_list)
        agents = []

        for i in range(n_veh):
            for j in range(n_des):
                agents.append(Agent(state_dim, action_dim))
                path = os.path.join(model_dir, 'agent-%d-%d.m' % (i, j))
                agents[-1].model.load_state_dict(torch.load(path))

        V2I_capacity_list = []
        probability = []

        with torch.no_grad():
            for e in range(test_episode):
                env.renew_environment()
                env.reset_payload(payload_size)
                env.reset_time(time_slow)
                env.is_active = np.ones((n_veh, n_des), dtype='bool')
                env.compute_V2V_interference(np.zeros((n_veh, n_des, 2), dtype='int32'))
                for t in range(steps_per_episode):
                    actions = np.zeros((n_veh, n_des, 2), dtype='int32')
                    for i in range(n_veh):
                        for j in range(n_des):
                            # 获取当前状态
                            idx = i * n_des + j
                            # 即完成训练的episode以及epsilon
                            state = get_local_state(env, (i, j))
                            epsilon_greedy_action = agents[idx].predict(state, final_epsilon)
                            actions[i, j, 0] = epsilon_greedy_action % n_RB
                            actions[i, j, 1] = int(epsilon_greedy_action / n_RB)
                    actions__ = actions.copy()
                    r_t, V2I_rate = env.act_in_env(actions__)
                    V2I_capacity_list.append(V2I_rate.sum() * env.bandwidth)
                    env.renew_channels_fastfading()
                    env.compute_V2V_interference(actions__)
                probability.append(np.count_nonzero(env.remain_payload <= 0) / (n_veh * n_des))
            V2I_capacity_avg = sum(V2I_capacity_list) / len(V2I_capacity_list)
            p_avg = sum(probability) / len(probability)
            print('平均总V2I信道容量: ', V2I_capacity_avg)
            print('平均载荷传输成功率: ', p_avg)
        return V2I_capacity_avg, p_avg

    def test_model_robustness(self, model_dir, N_list: list):
        V2I_output = []
        p_output = []
        payload_size_list = [1060 * N for N in N_list]
        for B in payload_size_list:
            V2I, p = self.test_model(model_dir, B)
            V2I_output.append(V2I)
            p_output.append(p)

        fig = plt.figure()
        ax1 = fig.add_subplot(222)
        ax1.set(title='V2I capacity', ylabel='V2V Capacity Sum(Mbps)', xlabel='payload size (Bytes)')
        ax1.plot(N_list, V2I_output)

        ax4 = fig.add_subplot(224)
        ax4.set(title='V2V payload transmission probability', ylabel='Transmission Probability',
                xlabel='payload size (Bytes)')
        ax4.plot(N_list, p_output)
        plt.show()

    def test_model_with_baseline(self, model_dir, payload_size):
        env = Environment.Environment(n_veh, n_des, payload_size)
        env.new_random_game()
        test_episode = 100
        final_epsilon = 0.02
        steps_per_episode = int(time_slow / time_fast)
        state_dim = len(get_local_state(env))
        action_dim = n_RB * len(env.V2V_power_dB_list)
        agents = []
        env.compute_V2V_interference(np.zeros((n_veh, n_des, 2), dtype='int32'))
        for i in range(n_veh):
            for j in range(n_des):
                agents.append(Agent(state_dim, action_dim))
                path = os.path.join(model_dir, 'agent-%d-%d.m' % (i, j))
                agents[-1].model.load_state_dict(torch.load(path))

        V2I_capacity_list = []
        probability = []

        V2I_capacity_list_rand = []
        probability_rand = []

        with torch.no_grad():
            for e in range(test_episode):
                env.renew_environment()
                # MARL
                env.reset_payload(payload_size)
                env.reset_time(time_slow)
                env.is_active = np.ones((n_veh, n_des), dtype='bool')
                # RandomBaseline
                env.remain_payload_rand = np.full((n_veh, n_des), payload_size)
                env.remain_time_rand = time_slow
                env.is_active_rand = np.ones((n_veh, n_des), dtype='bool')
                env.compute_V2V_interference(np.zeros((n_veh, n_des, 2), dtype='int32'))
                for t in range(steps_per_episode):
                    # MARL
                    actions = np.zeros((n_veh, n_des, 2), dtype='int32')
                    for i in range(n_veh):
                        for j in range(n_des):
                            # 获取当前状态
                            idx = i * n_des + j
                            # 即完成训练的episode以及epsilon
                            state = get_local_state(env, (i, j))
                            epsilon_greedy_action = agents[idx].predict(state, final_epsilon)
                            actions[i, j, 0] = epsilon_greedy_action % n_RB
                            actions[i, j, 1] = int(epsilon_greedy_action / n_RB)
                            print(actions[i, j])
                    actions__ = actions.copy()
                    r_t, V2I_rate = env.act_in_env(actions__)
                    V2I_capacity_list.append(V2I_rate.sum() * env.bandwidth)

                    # RandomBaseline
                    actions_rand = np.zeros((n_veh, n_des, 2), dtype='int32')
                    for i in range(n_veh):
                        for j in range(n_des):
                            action_rand = np.random.randint(0, len(V2V_power_dB_List) * n_RB)
                            actions_rand[i, j, 0] = action_rand % n_RB
                            actions_rand[i, j, 1] = action_rand // n_RB
                    reward, V2I_rate = env.act_in_env_for_rand(actions_rand)
                    V2I_capacity_list_rand.append(V2I_rate.sum() * env.bandwidth)
                    env.renew_channels_fastfading()
                    env.compute_V2V_interference(actions__)
                p = np.count_nonzero(env.remain_payload <= 0) / (n_veh * n_des)
                probability.append(p)
                p_rand = np.count_nonzero(env.remain_payload_rand <= 0) / (n_veh * n_des)
                probability_rand.append(p_rand)
                print('MARL:', p, 'RandomBaseline:', p_rand)
            V2I_capacity_avg = sum(V2I_capacity_list) / len(V2I_capacity_list)
            p_avg = sum(probability) / len(probability)
            print("MARL:")
            print('平均总V2I信道容量: ', V2I_capacity_avg)
            print('平均载荷传输成功率: ', p_avg)
            print()
            print("RandomBaseline:")
            print('平均总V2I信道容量: ', np.mean(V2I_capacity_list_rand))
            print('平均载荷传输成功率: ', np.mean(probability_rand))


model_dir = 'models v5.0'
al = Algorithm()
al.train_model(model_dir)
al.test_model(model_dir, 2 * 1060)
# al.test_model_with_baseline(model_dir, 2 * 1060)
# al.test_model_robustness('models v2.0', [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6])
