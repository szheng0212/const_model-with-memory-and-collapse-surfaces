# -*- coding: utf-8 -*-
"""
Created on Mon May  3 19:06:02 2021

@author: SZheng
"""

# from function_tools import Mc_bracket
import numpy as np

# def NewtonIteration(a1,a2,a3,a4,c1,c2,c3,c4,r1,r2,r3,r4,G,sgm0,Q1,R1,bios1,\
#                     Q2,R2,bios2,R3,bios3,stress_trial):
def NewtonIteration(ai, ci, ri, G, sgm0, Qi, Ri, R3, biosi, bios3, stress_trial):
        """
        通过Newton-Raphson迭代，求解该塑性加载步中产生的等效塑性应变.
        当判断出材料在该加载步中处于塑性流动状态时，则根据上一步加载步求解结果的背应力a1，a2，a3，
        弹性域大小R及各向同性硬化饱和值Q，以及该加载步中的试应力值，通过迭代求解dp

        Args:   param1,2,3  a1, a2, a3    第n步加载步求解得到的 三个背应力分量
                param4      R             第n步加载步求解得到的 弹性域大小
                param5      stress_trial  第n+1步加载步的 试应力
        """
        stress_trial_n1 = stress_trial
        # 初始化迭代需要的变量值
        a_array_n = ai
        R_array_n = Ri
        R3_n = R3

        c_array = ci
        r_array = ri
        b_array = biosi
        Q_array = Qi

        sum_a_n_trial = np.sum(a_array_n)
        sum_R_n_trail = np.sum(R_array_n) + R3_n
        # 设置第n+1步的屈服状态方程,保证第n+1步结束后应力点位于屈服面上/内
        F_yield = abs(stress_trial_n1 - sum_a_n_trial) - (sgm0 + sum_R_n_trail)

        dp0 = 0
        dp = 0.1
        theta_array_n = 1 / (1 + r_array * dp0)

        a_theta_n = np.sum(a_array_n * theta_array_n)
        c_theta_n = np.sum(c_array * theta_array_n)
        # theta2_r_a_n = np.sum(theta_array_n ** 2 * r_array * a_array_n)
        # theta2_r_c_n = np.sum(theta_array_n ** 2 * r_array * c_array)

        sum_R_n1 = sum_R_n_trail
        R_array_n1 = R_array_n
        a_theta_n1 = a_theta_n
        c_theta_n1 = c_theta_n
        # theta2_r_a_n1 = theta2_r_a_n
        # theta2_r_c_n1 = theta2_r_c_n

        a = abs(dp - dp0)
        err = 0.000001
        err_l = -0.000001
        err_a = 0.000001
        # ERR_l = -1
        # ERR_1 = 0.00001
        err_dp = 0

        while(((F_yield > err) or (F_yield < err_l)) or (dp <= err_dp) or (a > err_a)):

            # 采用迭代方法直接求解dpn+1
            dp = (abs(stress_trial_n1 - a_theta_n1) - (sgm0 + sum_R_n1)) / (3 * G + c_theta_n1)
            a = abs(dp - dp0)
            # 计算θn+1,Rn+1
            theta_array_n1 = 1 / (1 + r_array * dp)

            a_theta_n1 = np.sum(a_array_n * theta_array_n1)
            c_theta_n1 = np.sum(c_array * theta_array_n1)
            # theta2_r_a_n1 = np.sum(theta_array_n1 ** 2 * r_array * a_array_n)
            # theta2_r_c_n1 = np.sum(theta_array_n1 ** 2 * r_array * c_array)
            R_array_n1 = (R_array_n + b_array * Q_array * dp) / (1 + b_array * dp)
            R3_n1 = R3_n + bios3 * dp
            sum_R_n1 = np.sum(R_array_n1) + R3_n1
            F_yield = abs(stress_trial_n1 - a_theta_n1) - (sgm0 + sum_R_n1) - (3 * G + c_theta_n1) * dp
            # if ((F_yield > err) and (F_yield < err_l)) and (dp <= err_dp) and (a >= err_a):
            #     break
            # else:
            #     dp0 = dp
            dp0 = dp


        if (dp <= 0):
            print('********dp不能小于等于0!*********')
            raise ValueError

        return dp              # 返回方程的解dp，即为该加载步的等效塑性应变

if __name__ == '__main__':
    l=NewtonIteration()
    print(l)
