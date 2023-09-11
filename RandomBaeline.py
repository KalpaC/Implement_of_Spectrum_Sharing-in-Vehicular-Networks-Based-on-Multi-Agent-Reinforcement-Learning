# RandomBaeline 2023/5/19 5:53

import numpy as np
import Environment

V2V_power_dB_List = [23, 10, 5, -100]
n_veh = 4
n_des = 1
n_RB = 4
time_slow = 0.1
time_fast = 0.001
payload_size = 2 * 1060
test_episodes = 100
steps_per_episode = int(time_slow / time_fast)

env = Environment.Environment(n_veh, n_des, payload_size)
env.new_random_game()

p_list = []
for e in range(test_episodes):
    env.renew_environment()
    env.reset_payload(payload_size)
    env.reset_time(time_slow)
    env.is_active = np.ones((n_veh, n_des), dtype='bool')
    for t in range(steps_per_episode):
        actions = np.zeros((n_veh, n_des, 2), dtype='int32')
        for i in range(n_veh):
            for j in range(n_des):
                action = np.random.randint(0, len(V2V_power_dB_List) * n_RB)
                actions[i, j, 0] = action % n_RB
                actions[i, j, 1] = action // n_RB
        action__ = actions.copy()
        reward, V2I_rate = env.act_in_env(action__)
        env.compute_V2V_interference(action__)
    p = np.count_nonzero(env.remain_payload <= 0) / (n_veh * n_des)
    p_list.append(p)
    if e % 10 == 9:
        print('平均p值为:', np.mean(p_list))
        p_list = []

