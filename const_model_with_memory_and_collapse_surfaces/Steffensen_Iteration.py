'''
Author: your name
Date: 2021-10-04 22:11:59
LastEditTime: 2021-10-06 22:31:37
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \Constitutive_Model_parameter_optimization_version0908\Steffensen_Iteration.py
'''
import numpy as np
from Heaviside import Heaviside
from Heaviside import Mac_bracket
# 给定一个 非零的 塑性应变增量 dp_n1和对应的θ_n1_0初值,通过迭代和隐式依赖链式关系进行求解,得到一个新的θ_n1_0值
def theta_iteration(sgm0, theta_n1, dp_n1, coefficient, stress_trial, a_array_n, c_array_n, bt_array, b_array, cs_array, h_array, Q_array_n,Qi_s_array, R_array_n, G, peps_n, kx_n, q_n, q_u_n, peps_un, yt):
    # 计算在经过迭代后 更新的 dp_n1值 来更新该试算步和试算状态下的 塑性流动方向
    # coefficient = np.array(1 + f_n1 * r_array * dp_n1)
    direction_n1_trial = np.sign((theta_n1 / (1 + 3 * G * dp_n1 * theta_n1)) * (stress_trial - np.sum(a_array_n / coefficient)))
    # 更新试状态 的塑性应变增量和总塑性应变
    dpeps = direction_n1_trial * dp_n1
    # 计算当前的 试塑性应变peps_n1
    peps_n1 = peps_n + dpeps

    # 验证当前的试算peps_n1是否在记忆面和溃散面内，调用Heaviside函数
    # 第n步的记忆面模型信息，为确定性状态
    location_g = kx_n
    radius_g = q_n

    # 第n步的崩溃目标面模型信息，为实时更新的确定性状态
    radius_c = q_u_n
    location_c = (peps_n + peps_un) / 2

    # 试算该试计算状态下的试塑性应变与第n步的记忆面模型,崩溃目标面模型的关系
    index_g_n1_trial = Heaviside(location_g, radius_g, peps_n1)
    index_c_n1_trial = Heaviside(location_c, radius_c, peps_n1)
    # 计算第n步的状态与记忆面模型的关系
    index_g_n = Heaviside(location_g, radius_g, peps_n)
    # 另外计算第n步的记忆面状态

    if index_c_n1_trial > 0:
    # 此时 试塑性应变 有向崩溃面 外部运动 或在超球面上 的趋势,不激活崩溃效应,对应的是index_c = 1或2
    # 使用相应的 扩张的演化公式进行计算
        if index_g_n1_trial > 0:
        # 在记忆面外，记忆面 发生扩张且移动，且崩溃面需要实时更新
            # 计算当前dp_n1对应的 塑性应变增量方向
            gama_dp_n1 = Mac_bracket(peps_n1, kx_n, q_n)
            dq_n1 = yt * gama_dp_n1
            dkx_n1 = (1 - yt) * ((peps_n1 - kx_n) / (q_n + gama_dp_n1)) * gama_dp_n1

            q_n1 = q_n + dq_n1
            kx_n1 = kx_n + dkx_n1
            # 根据更新的记忆面更新系数C和Q_n1
            c_array_n1 = (c_array_n + h_array * cs_array * dq_n1) / (1 + h_array * dq_n1)
            Q_array_n1 = (Q_array_n + bt_array * Qi_s_array * dq_n1) / (1 + bt_array * dq_n1)
            # 更新R_n1
            R_array_n1 = (R_array_n + b_array * Q_array_n1 * dp_n1) / (1 + b_array * dp_n1)
            # 更新θ_n1
            theta_n1_1 = 1 / (sgm0 + np.sum(R_array_n1) + np.sum((c_array_n1 * dp_n1) / coefficient))
            return theta_n1_1
        else:
            # index_g_n1_trial==0,下一步在面内,且不激活溃散效应
            # 在该情况下,记忆面模型不变化,溃散面模型实时更新
            gama_dp_n1 = Mac_bracket(peps_n1, kx_n, q_n)
            dq_n1 = yt * gama_dp_n1
            dkx_n1 = (1 - yt) * ((peps_n1 - kx_n) / (q_n + gama_dp_n1)) * gama_dp_n1

            q_n1 = q_n + dq_n1
            kx_n1 = kx_n + dkx_n1
            # 根据更新的记忆面更新系数C和Q_n1
            c_array_n1 = (c_array_n + h_array * cs_array * dq_n1) / (1 + h_array * dq_n1)
            Q_array_n1 = (Q_array_n + bt_array * Qi_s_array * dq_n1) / (1 + bt_array * dq_n1)
            # 更新R_n1
            R_array_n1 = (R_array_n + b_array * Q_array_n1 * dp_n1) / (1 + b_array * dp_n1)
            # 更新θ_n1
            theta_n1_1 = 1 / (sgm0 + np.sum(R_array_n1) + np.sum((c_array_n1 * dp_n1) / coefficient))
            # 此时，θ_n1仍发生了变化，主要是dp_n1引起的R_n1的变化，造成了θ的变化
            return theta_n1_1
    elif index_c_n1_trial == 0:
        # 溃散目标面c有向内运动的趋势,此时激活消散效应
        if index_g_n == 0:


