# 作者 Ajex
# 创建时间 2023/5/5 19:11
# 文件名 Environment.py
import random
import time
import numpy as np
import math

from typing import List

def power_dB2W(power_dB):
    return 10 ** (power_dB / 10)


class Vehicle:
    # Vehicle simulator: include all the information for a vehicle
    def __init__(self, start_position, start_direction, velocity):
        self.position = start_position
        self.direction = start_direction
        self.velocity = velocity
        self.destinations = []
        self.turn = ''


class V2V_Calculator:
    def __init__(self):
        self.h_bs = 1.5
        self.h_ms = 1.5
        self.fc = 2
        self.decorrelation_distance = 10
        self.shadow_std = 3

    def get_path_loss(self, position_A, position_B):
        d1 = abs(position_A[0] - position_B[0])
        d2 = abs(position_A[1] - position_B[1])
        d = math.hypot(d1, d2) + 0.001
        d_bp = 4 * (self.h_bs - 1) * (self.h_ms - 1) * self.fc * (10 ** 9) / (3 * 10 ** 8)

        def PL_Los(d):
            if d <= 3:
                return 22.7 * np.log10(3) + 41 + 20 * np.log10(self.fc / 5)
            else:
                if d < d_bp:
                    return 22.7 * np.log10(d) + 41 + 20 * np.log10(self.fc / 5)
                else:
                    return 40.0 * np.log10(d) + 9.45 - 17.3 * np.log10(self.h_bs) - 17.3 * np.log10(
                        self.h_ms) + 2.7 * np.log10(self.fc / 5)

        def PL_NLos(d_a, d_b):
            n_j = max(2.8 - 0.0024 * d_b, 1.84)
            return PL_Los(d_a) + 20 - 12.5 * n_j + 10 * n_j * np.log10(d_b) + 3 * np.log10(self.fc / 5)

        if min(d1, d2) < 7:
            PL = PL_Los(d)
        else:
            PL = min(PL_NLos(d1, d2), PL_NLos(d2, d1))
        return PL

    def get_shadowing(self, delta_distance, shadowing):
        return np.exp(-1 * (delta_distance / self.decorrelation_distance)) * shadowing \
               + math.sqrt(1 - np.exp(-2 * (delta_distance / self.decorrelation_distance))) * np.random.normal(0, 3)


class V2I_Calculator:
    def __init__(self):
        self.h_bs = 25
        self.h_ms = 1.5
        self.Decorrelation_distance = 50
        self.BS_position = [750 / 4, 1299 / 4]  # center of the grids
        self.shadow_std = 8

    def get_path_loss(self, position_A):
        d1 = abs(position_A[0] - self.BS_position[0])
        d2 = abs(position_A[1] - self.BS_position[1])
        distance = math.hypot(d1, d2)
        return 128.1 + 37.6 * np.log10(
            math.sqrt(distance ** 2 + (self.h_bs - self.h_ms) ** 2) / 1000)  # + self.shadow_std * np.random.normal()

    def get_shadowing(self, delta_distance, shadowing):
        return np.multiply(np.exp(-1 * (delta_distance / self.Decorrelation_distance)), shadowing) \
               + np.sqrt(1 - np.exp(-2 * (delta_distance / self.Decorrelation_distance))) \
               * np.random.normal(0, self.shadow_std, len(shadowing))


class Environment:
    up_lanes = [3.5 / 2, 3.5 / 2 + 3.5, 250 + 3.5 / 2, 250 + 3.5 + 3.5 / 2, 500 + 3.5 / 2, 500 + 3.5 + 3.5 / 2]
    down_lanes = [250 - 3.5 - 3.5 / 2, 250 - 3.5 / 2, 500 - 3.5 - 3.5 / 2, 500 - 3.5 / 2, 750 - 3.5 - 3.5 / 2,
                  750 - 3.5 / 2]
    left_lanes = [3.5 / 2, 3.5 / 2 + 3.5, 433 + 3.5 / 2, 433 + 3.5 + 3.5 / 2, 866 + 3.5 / 2, 866 + 3.5 + 3.5 / 2]
    right_lanes = [433 - 3.5 - 3.5 / 2, 433 - 3.5 / 2, 866 - 3.5 - 3.5 / 2, 866 - 3.5 / 2, 1299 - 3.5 - 3.5 / 2,
                   1299 - 3.5 / 2]
    for i in range(len(up_lanes)):
        up_lanes[i] = up_lanes[i] / 2
        down_lanes[i] = down_lanes[i] / 2
        left_lanes[i] = left_lanes[i] / 2
        right_lanes[i] = right_lanes[i] / 2

    lanes = {'u': up_lanes, 'd': down_lanes, 'l': left_lanes, 'r': right_lanes}
    width = 750 / 2
    height = 1299 / 2

    def __init__(self, M, K, B):
        """
        尊重论文符号的参数列表，以及使用符合论文设定的默认环境参数值。
        :param M: 在本环境下等于子载波数、车辆数、V2I link数
        :param K: 每辆车的“邻居”个数，即每辆车与多少辆其他车建立V2V link
        :param B: 场景初始时所有V2V link的有效载荷数量
        :return: None
        """
        # 固定的环境参数或变量
        self.V2I_power_dB = 23
        self.V2V_power_dB_list = [23, 10, 5, -100]
        sig2_dB = -114  # dB
        self.bsAntGain = 8  # dB
        self.bsNoiseFigure = 5  # dB
        self.vehAntGain = 3  # dB
        self.vehNoiseFigure = 9  # dB
        self.sig2 = power_dB2W(sig2_dB)

        self.n_veh = M
        self.n_RB = M
        self.n_des = K
        self.time_fast = 0.001  # s
        self.time_slow = 0.1
        self.bandwidth = int(1e6)  # MHz
        self.payload_size = B * 8  # bytes

        self.vehicles: List[Vehicle] = []
        self.remain_time = self.time_slow  # s

        # 与V2V有关而可能会改变的部分
        # self.beta = 8 * 1000 * 1000  # 尚未确定

        self.V2V = V2V_Calculator()
        self.V2I = V2I_Calculator()
        self.V2I_channels_with_fastfading = None
        self.V2V_channels_with_fastfading = None
        self.is_active = np.ones((self.n_veh, self.n_des),dtype='bool')
        self.remain_payload = np.full((self.n_veh, self.n_des), self.payload_size)
        self.V2V_Interference_dB = np.zeros((self.n_veh, self.n_des, self.n_RB)) + sig2_dB
        # 随机基线
        self.is_active_rand = np.ones((self.n_veh, self.n_des),dtype='bool')
        self.remain_payload_rand = np.full((self.n_veh, self.n_des), self.payload_size)
        self.remain_time_rand = self.time_slow




    def new_random_game(self):
        self.init_all_vehicles()
        # 更新信道
        self.reset_payload(self.payload_size)
        self.reset_time(self.time_slow)
        self.is_active = np.ones((self.n_veh, self.n_des), dtype='bool')
        self.V2V_Shadowing = np.random.normal(0, self.V2V.shadow_std, (self.n_veh, self.n_veh))
        self.V2I_Shadowing = np.random.normal(0, self.V2I.shadow_std, self.n_veh)
        self.renew_channel()
        self.renew_channels_fastfading()

    def reset_payload(self, B):
        self.payload_size = B * 8
        self.remain_payload = np.full((self.n_veh, self.n_des), self.payload_size)

    def reset_time(self, T):
        self.remain_time = T

    def add_new_vehicle(self):
        # 目前所有车辆的速度均为36km/h，如果需要修改或引入随机性，请重新设置
        direction = np.random.choice(['u', 'd', 'l', 'r'])
        road = np.random.randint(0, len(self.lanes[direction]))
        if direction == 'u' or direction == 'd':
            x = self.lanes[direction][road]
            y = np.random.randint(3.5 / 2, self.height - 3.5 / 2)
        else:
            x = np.random.randint(3.5 / 2, self.width - 3.5 / 2)
            y = self.lanes[direction][road]
        position = [x, y]
        self.vehicles.append(Vehicle(position, direction, np.random.randint(10, 15)))

    def init_all_vehicles(self):
        # 初始化全部的vehicle，
        for i in range(self.n_veh):
            self.add_new_vehicle()
        self.get_destination()
        self.delta_distance = np.array([c.velocity * self.time_slow for c in self.vehicles])

    def get_destination(self):
        # 找到对每辆车找到距离它最近的self.n_des辆车
        # 每次更新位置之后都需要重新判断，因为数据包的有效期恰好也过了
        positions = np.array([c.position for c in self.vehicles])
        distance = np.zeros((self.n_veh, self.n_veh))
        for i in range(self.n_veh):
            for j in range(self.n_veh):
                # np.linalg.norm用于计算向量的模，此处可以用于计算两点间距离
                distance[i][j] = np.linalg.norm(positions[i] - positions[j])
        for i in range(self.n_veh):
            sort_idx = np.argsort(distance[:, i])
            self.vehicles[i].destinations = sort_idx[1:1 + self.n_des]

    def renew_channel(self):
        self.V2V_pathloss = np.zeros((self.n_veh, self.n_veh)) + 50 * np.identity(self.n_veh)
        self.V2I_pathloss = np.zeros(self.n_veh)
        # self.V2V_channels_abs = np.zeros((self.n_veh, self.n_veh))
        # self.V2I_channels_abs = np.zeros(self.n_veh)
        for i in range(self.n_veh):
            for j in range(i + 1, self.n_veh):
                self.V2V_Shadowing[j][i] = self.V2V_Shadowing[i][j] = self.V2V.get_shadowing(
                    self.delta_distance[i] + self.delta_distance[j], self.V2V_Shadowing[i][j])
                self.V2V_pathloss[j, i] = self.V2V_pathloss[i][j] = self.V2V.get_path_loss(
                    self.vehicles[i].position, self.vehicles[j].position)

        self.V2V_channels_abs = self.V2V_pathloss + self.V2V_Shadowing
        self.V2I_Shadowing = self.V2I.get_shadowing(self.delta_distance, self.V2I_Shadowing)
        for i in range(len(self.vehicles)):
            self.V2I_pathloss[i] = self.V2I.get_path_loss(self.vehicles[i].position)
        self.V2I_channels_abs = self.V2I_pathloss + self.V2I_Shadowing

    def renew_channels_fastfading(self):
        # 计算并更新得到新的包含fastfading的信道信息
        V2V_channels_with_fastfading = np.repeat(self.V2V_channels_abs[:, :, np.newaxis], self.n_RB, axis=2)
        self.V2V_channels_with_fastfading = V2V_channels_with_fastfading - 20 * np.log10(
            np.abs(np.random.normal(0, 1, V2V_channels_with_fastfading.shape)
                   + 1j * np.random.normal(0, 1, V2V_channels_with_fastfading.shape)) / math.sqrt(2))

        V2I_channels_with_fastfading = np.repeat(self.V2I_channels_abs[:, np.newaxis], self.n_RB, axis=1)
        self.V2I_channels_with_fastfading = V2I_channels_with_fastfading - 20 * np.log10(
            np.abs(np.random.normal(0, 1, V2I_channels_with_fastfading.shape)
                   + 1j * np.random.normal(0, 1, V2I_channels_with_fastfading.shape)) / math.sqrt(2))

    def renew_environment(self):
        # 执行周期维100ms的统一更新，所以并不包括FastFading的更新
        self.renew_positions()
        self.get_destination()
        self.renew_channel()
        self.renew_channels_fastfading()

    def compute_V2V_interference(self, all_actions):
        all_actions = all_actions.copy()
        V2V_interference = np.zeros((self.n_veh, self.n_des, self.n_RB)) + self.sig2
        channel = all_actions[:, :, 0]
        power_selection = all_actions[:, :, 1]
        channel[np.logical_not(self.is_active)] = -1
        # 计算V2I的干扰
        for m in range(self.n_RB):
            for i in range(self.n_veh):
                for j in range(self.n_des):
                    V2V_interference += power_dB2W(
                        self.V2I_power_dB
                        + self.interference_dB_from_V2I_to_V2V((i, j), m)
                    )
        for i in range(self.n_veh):
            for j in range(self.n_des):
                for k in range(self.n_veh):
                    for l in range(self.n_des):
                        if i == k and j == l or channel[i, j] < 0:
                            # 车号相同或不再激活
                            pass
                        else:
                            V2V_interference[k, l, channel[i][j]] += power_dB2W(
                                self.V2V_power_dB_list[power_selection[i, j]]
                                + self.interference_dB_from_other_V2V((k, l), (i, j), channel[i][j])
                            )
        self.V2V_Interference_dB = 10 * np.log10(V2V_interference)
        return self.V2V_Interference_dB



    def gain_dB_of_V2V(self, V2V_Channel, m):
        """
        g_k[m] V2V link k的发射器到接收器之间的信道增益dB
        :param V2V_Channel: (veh_id, des_id)
        :param m: 子载波编号m
        :return: 信道增益dB，正值则为增益
        """
        sent = V2V_Channel[0]
        recv = self.vehicles[sent].destinations[V2V_Channel[1]]
        minus = self.V2V_channels_with_fastfading[sent][recv][m] + self.vehNoiseFigure
        plus = self.vehAntGain * 2
        return -minus + plus

    def gain_dB_of_V2I(self, m):
        """
        g_{m,B}[m]
        :param m: V2I link m，子载波编号m（该环境默认二者相同）
        :return: 信道增益dB，正值则为增益
        """
        sent = m
        minus = self.V2I_channels_with_fastfading[m][m] + self.bsNoiseFigure
        plus = self.vehAntGain + self.bsAntGain
        return -minus + plus

    def interference_dB_from_other_V2V(self, V2V_Channel, other_V2V, m):
        """
        g_{k',k}[m] 从其他V2V k'的发射器到本V2V k的接收器的干扰功率增益dB
        :param V2V_Channel: (veh_id, des_id)
        :param other_V2V: (veh_id, des_id)
        :param m: 子载波编号m
        :return: 信道增益dB，正值则为增益
        """
        sent = other_V2V[0]
        recv = self.vehicles[V2V_Channel[0]].destinations[V2V_Channel[1]]
        minus = self.V2V_channels_with_fastfading[sent][recv][m] + self.vehNoiseFigure
        plus = self.vehAntGain * 2
        return -minus + plus

    def interference_dB_from_V2V_to_BS(self, V2V_Channel, m):
        """
        g_{k,B}[m] 从V2V link k的发射器到基站B之间的干扰功率增益dB
        :param V2V_Channel: (veh_id, des_id)
        :param m: 子载波编号m
        :return: 信道增益dB，正值则为增益
        """
        sent = V2V_Channel[0]
        minus = self.V2I_channels_with_fastfading[sent][m] + self.bsNoiseFigure
        plus = self.vehAntGain + self.bsAntGain
        return -minus + plus

    def interference_dB_from_V2I_to_V2V(self, V2V_Channel, m):
        """
        \\hat{g}_{m,k}[m] 从V2I link的发射器m到V2V link k的接收器之间的干扰功率增益dB
        :param V2V_Channel: (veh_id, des_id)
        :param m: 子载波编号m，也是V2I link m
        :return: 信道增益dB，正值则为增益
        """
        sent = m
        recv = self.vehicles[V2V_Channel[0]].destinations[V2V_Channel[1]]
        minus = self.V2V_channels_with_fastfading[sent][recv][m] + self.vehNoiseFigure
        plus = self.vehAntGain * 2
        return -minus + plus

    def act_in_env(self, all_actions):
        # 该版本非个人实现
        all_actions = all_actions.copy()
        actions = all_actions[:, :, 0]  # the channel_selection_part
        power_selection = all_actions[:, :, 1]  # power selection

        V2I_Interference = np.zeros(self.n_RB)  # V2I interference
        for i in range(len(self.vehicles)):
            for j in range(self.n_des):
                if not self.is_active[i, j]:
                    continue
                V2I_Interference[actions[i][j]] += 10 ** ((self.V2V_power_dB_list[power_selection[i, j]] -
                                                           self.V2I_channels_with_fastfading[i, actions[i, j]]
                                                           + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)
        self.V2I_Interference = V2I_Interference + self.sig2
        V2I_Signals = 10 ** ((
                                     self.V2I_power_dB - self.V2I_channels_with_fastfading.diagonal() + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)
        V2I_Rate = np.log2(1 + np.divide(V2I_Signals, self.V2I_Interference))


        V2V_Interference = np.zeros((len(self.vehicles), self.n_des))
        V2V_Signal = np.zeros((len(self.vehicles), self.n_des))
        actions[(np.logical_not(
            self.is_active))] = -1
        for i in range(self.n_RB):
            indexes = np.argwhere(actions == i)
            for j in range(len(indexes)):
                receiver_j = self.vehicles[indexes[j, 0]].destinations[indexes[j, 1]]
                V2V_Signal[indexes[j, 0], indexes[j, 1]] = 10 ** (
                        (self.V2V_power_dB_list[power_selection[indexes[j, 0], indexes[j, 1]]]
                         - self.V2V_channels_with_fastfading[
                             indexes[j][0], receiver_j, i] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)

                V2V_Interference[indexes[j, 0], indexes[j, 1]] += 10 ** ((self.V2I_power_dB -
                                                                          self.V2V_channels_with_fastfading[
                                                                              i, receiver_j, i] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)

                for k in range(j + 1, len(indexes)):  # spectrum-sharing V2Vs
                    receiver_k = self.vehicles[indexes[k][0]].destinations[indexes[k][1]]
                    V2V_Interference[indexes[j, 0], indexes[j, 1]] += 10 ** (
                            (self.V2V_power_dB_list[power_selection[indexes[k, 0], indexes[k, 1]]]
                             - self.V2V_channels_with_fastfading[indexes[k][0]][receiver_j][
                                 i] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)
                    V2V_Interference[indexes[k, 0], indexes[k, 1]] += 10 ** (
                            (self.V2V_power_dB_list[power_selection[indexes[j, 0], indexes[j, 1]]]
                             - self.V2V_channels_with_fastfading[indexes[j][0]][receiver_k][
                                 i] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)
        self.V2V_Interference = V2V_Interference + self.sig2
        V2V_Rate = np.log2(1 + np.divide(V2V_Signal, self.V2V_Interference))
        self.remain_payload -= (V2V_Rate * self.time_fast * self.bandwidth).astype('int32')
        self.remain_payload[self.remain_payload < 0] = 0
        self.remain_time -= self.time_fast

        reward_elements = V2V_Rate / 10
        reward_elements[self.remain_payload <= 0] = 1
        self.is_active[
            np.multiply(self.is_active, self.remain_payload <= 0)] = 0
        lambdda = 0.1
        reward = lambdda * np.sum(V2I_Rate) / (self.n_veh * 10) \
                 + (1 - lambdda) * np.sum(reward_elements) / (self.n_veh * self.n_des)
        return reward, V2I_Rate

    def act_in_env_for_rand(self,all_actions):
        # 该版本非个人实现
        all_actions = all_actions.copy()
        actions = all_actions[:, :, 0]  # the channel_selection_part
        power_selection = all_actions[:, :, 1]  # power selection

        V2I_Interference_rand = np.zeros(self.n_RB)  # V2I interference
        for i in range(len(self.vehicles)):
            for j in range(self.n_des):
                if not self.is_active_rand[i, j]:
                    continue
                V2I_Interference_rand[actions[i][j]] += 10 ** ((self.V2V_power_dB_list[power_selection[i, j]] -
                                                           self.V2I_channels_with_fastfading[i, actions[i, j]]
                                                           + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)
        self.V2I_Interference_rand = V2I_Interference_rand + self.sig2
        V2I_Signals = 10 ** ((
                                     self.V2I_power_dB - self.V2I_channels_with_fastfading.diagonal() + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)
        V2I_Rate = np.log2(1 + np.divide(V2I_Signals, self.V2I_Interference))

        V2V_Interference = np.zeros((len(self.vehicles), self.n_des))
        V2V_Signal = np.zeros((len(self.vehicles), self.n_des))
        actions[(np.logical_not(
            self.is_active))] = -1
        for i in range(self.n_RB):
            indexes = np.argwhere(actions == i)
            for j in range(len(indexes)):
                receiver_j = self.vehicles[indexes[j, 0]].destinations[indexes[j, 1]]
                V2V_Signal[indexes[j, 0], indexes[j, 1]] = 10 ** (
                        (self.V2V_power_dB_list[power_selection[indexes[j, 0], indexes[j, 1]]]
                         - self.V2V_channels_with_fastfading[
                             indexes[j][0], receiver_j, i] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)

                V2V_Interference[indexes[j, 0], indexes[j, 1]] += 10 ** ((self.V2I_power_dB -
                                                                          self.V2V_channels_with_fastfading[
                                                                              i, receiver_j, i] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)

                for k in range(j + 1, len(indexes)):  # spectrum-sharing V2Vs
                    receiver_k = self.vehicles[indexes[k][0]].destinations[indexes[k][1]]
                    V2V_Interference[indexes[j, 0], indexes[j, 1]] += 10 ** (
                            (self.V2V_power_dB_list[power_selection[indexes[k, 0], indexes[k, 1]]]
                             - self.V2V_channels_with_fastfading[indexes[k][0]][receiver_j][
                                 i] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)
                    V2V_Interference[indexes[k, 0], indexes[k, 1]] += 10 ** (
                            (self.V2V_power_dB_list[power_selection[indexes[j, 0], indexes[j, 1]]]
                             - self.V2V_channels_with_fastfading[indexes[j][0]][receiver_k][
                                 i] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)
        self.V2V_Interference = V2V_Interference + self.sig2
        V2V_Rate = np.log2(1 + np.divide(V2V_Signal, self.V2V_Interference))
        self.remain_payload_rand -= (V2V_Rate * self.time_fast * self.bandwidth).astype('int32')
        self.remain_payload_rand[self.remain_payload_rand < 0] = 0
        self.remain_time_rand -= self.time_fast

        reward_elements = V2V_Rate / 10
        reward_elements[self.remain_payload_rand <= 0] = 1
        self.is_active[
            np.multiply(self.is_active_rand, self.remain_payload_rand <= 0)] = 0
        lambdda = 0.1
        reward = lambdda * np.sum(V2I_Rate) / (self.n_veh * 10) \
                 + (1 - lambdda) * np.sum(reward_elements) / (self.n_veh * self.n_des)
        return reward, V2I_Rate


    def renew_positions(self):
        # 论文作者给出的位置更新函数，基于该函数可以得到与论文相对接近的结果
        i = 0
        while (i < len(self.vehicles)):
            delta_distance = self.vehicles[i].velocity * self.time_slow
            change_direction = False
            if self.vehicles[i].direction == 'u':
                # print ('len of position', len(self.position), i)
                for j in range(len(self.left_lanes)):

                    if (self.vehicles[i].position[1] <= self.left_lanes[j]) and (
                            (self.vehicles[i].position[1] + delta_distance) >= self.left_lanes[
                        j]):  # came to an cross
                        if (np.random.uniform(0, 1) < 0.4):
                            self.vehicles[i].position = [self.vehicles[i].position[0] - (
                                    delta_distance - (self.left_lanes[j] - self.vehicles[i].position[1])),
                                                         self.left_lanes[j]]
                            self.vehicles[i].direction = 'l'
                            change_direction = True
                            break
                if change_direction == False:
                    for j in range(len(self.right_lanes)):
                        if (self.vehicles[i].position[1] <= self.right_lanes[j]) and (
                                (self.vehicles[i].position[1] + delta_distance) >= self.right_lanes[j]):
                            if (np.random.uniform(0, 1) < 0.4):
                                self.vehicles[i].position = [self.vehicles[i].position[0] + (
                                        delta_distance + (self.right_lanes[j] - self.vehicles[i].position[1])),
                                                             self.right_lanes[j]]
                                self.vehicles[i].direction = 'r'
                                change_direction = True
                                break
                if change_direction == False:
                    self.vehicles[i].position[1] += delta_distance
            if (self.vehicles[i].direction == 'd') and (change_direction == False):
                # print ('len of position', len(self.position), i)
                for j in range(len(self.left_lanes)):
                    if (self.vehicles[i].position[1] >= self.left_lanes[j]) and (
                            (self.vehicles[i].position[1] - delta_distance) <= self.left_lanes[
                        j]):  # came to an cross
                        if (np.random.uniform(0, 1) < 0.4):
                            self.vehicles[i].position = [self.vehicles[i].position[0] - (
                                    delta_distance - (self.vehicles[i].position[1] - self.left_lanes[j])),
                                                         self.left_lanes[j]]
                            # print ('down with left', self.vehicles[i].position)
                            self.vehicles[i].direction = 'l'
                            change_direction = True
                            break
                if change_direction == False:
                    for j in range(len(self.right_lanes)):
                        if (self.vehicles[i].position[1] >= self.right_lanes[j]) and (
                                self.vehicles[i].position[1] - delta_distance <= self.right_lanes[j]):
                            if (np.random.uniform(0, 1) < 0.4):
                                self.vehicles[i].position = [self.vehicles[i].position[0] + (
                                        delta_distance + (self.vehicles[i].position[1] - self.right_lanes[j])),
                                                             self.right_lanes[j]]
                                # print ('down with right', self.vehicles[i].position)
                                self.vehicles[i].direction = 'r'
                                change_direction = True
                                break
                if change_direction == False:
                    self.vehicles[i].position[1] -= delta_distance
            if (self.vehicles[i].direction == 'r') and (change_direction == False):
                # print ('len of position', len(self.position), i)
                for j in range(len(self.up_lanes)):
                    if (self.vehicles[i].position[0] <= self.up_lanes[j]) and (
                            (self.vehicles[i].position[0] + delta_distance) >= self.up_lanes[
                        j]):  # came to an cross
                        if (np.random.uniform(0, 1) < 0.4):
                            self.vehicles[i].position = [self.up_lanes[j], self.vehicles[i].position[1] + (
                                    delta_distance - (self.up_lanes[j] - self.vehicles[i].position[0]))]
                            change_direction = True
                            self.vehicles[i].direction = 'u'
                            break
                if change_direction == False:
                    for j in range(len(self.down_lanes)):
                        if (self.vehicles[i].position[0] <= self.down_lanes[j]) and (
                                (self.vehicles[i].position[0] + delta_distance) >= self.down_lanes[j]):
                            if (np.random.uniform(0, 1) < 0.4):
                                self.vehicles[i].position = [self.down_lanes[j], self.vehicles[i].position[1] - (
                                        delta_distance - (self.down_lanes[j] - self.vehicles[i].position[0]))]
                                change_direction = True
                                self.vehicles[i].direction = 'd'
                                break
                if change_direction == False:
                    self.vehicles[i].position[0] += delta_distance
            if (self.vehicles[i].direction == 'l') and (change_direction == False):
                for j in range(len(self.up_lanes)):

                    if (self.vehicles[i].position[0] >= self.up_lanes[j]) and (
                            (self.vehicles[i].position[0] - delta_distance) <= self.up_lanes[
                        j]):  # came to an cross
                        if (np.random.uniform(0, 1) < 0.4):
                            self.vehicles[i].position = [self.up_lanes[j], self.vehicles[i].position[1] + (
                                    delta_distance - (self.vehicles[i].position[0] - self.up_lanes[j]))]
                            change_direction = True
                            self.vehicles[i].direction = 'u'
                            break
                if change_direction == False:
                    for j in range(len(self.down_lanes)):
                        if (self.vehicles[i].position[0] >= self.down_lanes[j]) and (
                                (self.vehicles[i].position[0] - delta_distance) <= self.down_lanes[j]):
                            if (np.random.uniform(0, 1) < 0.4):
                                self.vehicles[i].position = [self.down_lanes[j], self.vehicles[i].position[1] - (
                                        delta_distance - (self.vehicles[i].position[0] - self.down_lanes[j]))]
                                change_direction = True
                                self.vehicles[i].direction = 'd'
                                break
                    if change_direction == False:
                        self.vehicles[i].position[0] -= delta_distance

            # if it comes to an exit
            if (self.vehicles[i].position[0] < 0) or (self.vehicles[i].position[1] < 0) or (
                    self.vehicles[i].position[0] > self.width) or (self.vehicles[i].position[1] > self.height):
                # delete
                #    print ('delete ', self.position[i])
                if (self.vehicles[i].direction == 'u'):
                    self.vehicles[i].direction = 'r'
                    self.vehicles[i].position = [self.vehicles[i].position[0], self.right_lanes[-1]]
                else:
                    if (self.vehicles[i].direction == 'd'):
                        self.vehicles[i].direction = 'l'
                        self.vehicles[i].position = [self.vehicles[i].position[0], self.left_lanes[0]]
                    else:
                        if (self.vehicles[i].direction == 'l'):
                            self.vehicles[i].direction = 'u'
                            self.vehicles[i].position = [self.up_lanes[0], self.vehicles[i].position[1]]
                        else:
                            if (self.vehicles[i].direction == 'r'):
                                self.vehicles[i].direction = 'd'
                                self.vehicles[i].position = [self.down_lanes[-1], self.vehicles[i].position[1]]

            i += 1

    def __renew_positions(self):
        # 自行实现的位置更新函数，与论文作者的版本有较大差别
        # 不能直行的条件判断
        def cant_go_straight(car: Vehicle):
            if car.direction == 'up' and car.position[1] > self.left_lanes[-1]:
                return True
            elif car.direction == 'down' and car.position[1] < self.right_lanes[0]:
                return True
            elif car.direction == 'left' and car.position[0] < self.down_lanes[0]:
                return True
            elif car.direction == 'right' and car.position[0] > self.up_lanes[-1]:
                return True
            return False

        # 不能左转的条件判断
        def cant_turn_left(car: Vehicle):
            if cant_go_straight(car):
                return True
            elif car.direction == 'up' and (car.position[0] == self.up_lanes[0] or car.position[0] == self.up_lanes[1]):
                return True
            elif car.direction == 'down' and (
                    car.position[0] == self.down_lanes[-2] or car.position[0] == self.down_lanes[-1]):
                return True
            elif car.direction == 'left' and (
                    car.position[1] == self.left_lanes[0] or car.position[1] == self.left_lanes[1]):
                return True
            elif car.direction == 'right' and (
                    car.position[1] == self.right_lanes[-2] or car.position[1] == self.right_lanes[-1]):
                return True
            return False

        # 对每辆车
        for v in self.vehicles:
            if v.turn == '':
                r = random.uniform(0, 1)
                if cant_go_straight(v) and cant_turn_left(v):
                    v.turn = 'right'
                elif cant_turn_left(v):
                    if 0 <= r < 0.5:
                        v.turn = 'straight'
                    else:
                        v.turn = 'right'
                else:
                    if 0 <= r < 0.25:
                        v.turn = 'left'
                    elif 0.25 <= r < 0.5:
                        v.turn = 'right'
                    else:
                        v.turn = 'straight'
            # 计算出下一步的位置
            delta_distance = self.time_slow * v.velocity
            turn_case = False
            if v.direction == 'up':
                if v.turn == 'right':
                    for right in self.right_lanes:
                        if v.position[1] < right < v.position[1] + delta_distance:
                            turn_case = True
                            v.direction = 'right'
                            exceed = v.position[1] + delta_distance - right
                            v.position[0] += exceed
                            v.position[1] = right
                            break
                else:
                    for left in self.left_lanes:
                        if v.position[1] < left < v.position[1] + delta_distance:
                            turn_case = True
                            if v.turn == 'straight':
                                v.position[1] += delta_distance
                                break
                            v.direction = 'left'
                            exceed = v.position[1] + delta_distance - left
                            v.position[0] -= exceed
                            v.position[1] = left
                            break
                if not turn_case:
                    v.position[1] += delta_distance
            elif v.direction == 'down':
                if v.turn == 'right':
                    # 向下+右转=向左
                    for left in self.left_lanes:
                        if v.position[1] > left > v.position[1] - delta_distance:
                            turn_case = True
                            v.direction = 'left'
                            exceed = left - (v.position[1] - delta_distance)
                            v.position[0] -= exceed
                            v.position[1] = left
                            break
                else:
                    # 向下+左转=向右
                    for right in self.right_lanes:
                        if v.position[1] > right > v.position[1] - delta_distance:
                            turn_case = True
                            if v.turn == 'straight':
                                v.position[1] -= delta_distance
                                break
                            v.direction = 'right'
                            exceed = right - (v.position[1] - delta_distance)
                            v.position[0] += exceed
                            v.position[1] = right
                            break
                if not turn_case:
                    v.position[1] -= delta_distance
            elif v.direction == 'left':
                if v.turn == 'right':
                    # 左+右=上
                    for up in self.up_lanes:
                        if v.position[0] - delta_distance < up < v.position[0]:
                            turn_case = True
                            v.direction = 'up'
                            exceed = up - (v.position[0] - delta_distance)
                            v.position[1] += exceed
                            v.position[0] = up
                            break
                else:
                    # 左+左=下
                    for down in self.down_lanes:
                        if v.position[0] - delta_distance < down < v.position[0]:
                            turn_case = True
                            if v.turn == 'straight':
                                v.position[0] -= delta_distance
                                break
                            v.direction = 'down'
                            exceed = down - (v.position[0] - delta_distance)
                            v.position[1] -= exceed
                            v.position[0] = down
                            break
                if not turn_case:
                    v.position[0] -= delta_distance
            else:
                if v.turn == 'right':
                    # 右+右 = 下
                    for down in self.down_lanes:
                        if v.position[0] < down < v.position[0] + delta_distance:
                            turn_case = True
                            v.direction = 'down'
                            exceed = v.position[0] + delta_distance - down
                            v.position[1] -= exceed
                            v.position[0] = down
                            break
                else:
                    # 右+左=上
                    for up in self.up_lanes:
                        if v.position[0] < up < v.position[0] + delta_distance:
                            turn_case = True
                            if v.turn == 'straight':
                                v.position[0] += delta_distance
                                break
                            v.direction = 'up'
                            exceed = v.position[0] + delta_distance - up
                            v.position[1] += exceed
                            v.position[0] = up
                            break
                if not turn_case:
                    v.position[0] += delta_distance
            if turn_case:
                v.turn = ''
        # 目前成功更新了每辆车的位置
        # 是否需要同时修改fast_fading等数据？先不了，还没有考虑如何存储。

    def __repr__(self):
        vehicle_info = [(v.position, v.direction, v.turn) for v in self.vehicles]
        return "Vehicle information:\n" + str(vehicle_info)
