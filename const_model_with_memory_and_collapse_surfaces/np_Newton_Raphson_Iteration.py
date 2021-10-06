# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 00:08:46 2021

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

        dp = 0     #0.00000015    ## 初值的选取对计算结果有很大影响  # 假定该步等效塑性应变值的初值为0   0.00015位每个加载步的平均应变增量值
        dp0 = 0
        # dp_l = []
        # dp_l.append(dp0)
        # F_l = []
        # d_F = []

        # dp = 0.1
        # a = dp - dp0
        ERR = 0.000001
        # ERR_l = -0.000001
        ERR_l = -1
        # ERR_1 = 0.00001
        ERR_dp = 0

        a_array_n = ai
        R_array_n = Ri
        R3_n = R3

        c_array = ci
        r_array = ri
        b_array = biosi
        Q_array = Qi

        sum_a_n_trial = np.sum(a_array_n)
        sum_R_n_trail = np.sum(R_array_n) + R3_n
        F_yield = abs(stress_trial - sum_a_n_trial) - (sgm0 + sum_R_n_trail)

        # F_l.append(F_yield)

        # 以dp = 0作为第n+1步的初值进行迭代计算
        # theta1 = 1 / (1 + r1 * dp0)  # 计算、更新各个θi值
        # theta2 = 1 / (1 + r2 * dp0)
        # theta3 = 1 / (1 + r3 * dp0)
        # theta4 = 1 / (1 + r4 * dp0)

        # theta_array = np.array([theta1,theta2,theta3,theta4])
        theta_array_n = 1 / (1 + r_array * dp0)

        a_theta_n = np.sum(a_array_n * theta_array_n)
        c_theta_n = np.sum(c_array * theta_array_n)
        theta2_r_a_n = np.sum(theta_array_n ** 2 * r_array * a_array_n)
        theta2_r_c_n = np.sum(theta_array_n ** 2 * r_array * c_array)
        # a_theta = a1 * theta1 + a2 * theta2 + a3 * theta3 + a4 * theta4
        # c_theta = theta1 * c1 + theta2 * c2 + theta3 * c3 + theta4 * c4
        # theta2_r_a = theta1**2 * r1 * a1 + theta2**2 * r2 * a2 + theta3**2 *r3 * a3 + theta4**2 *r4 * a4
        # theta2_r_c = theta1**2 * r1 * c1 + theta2**2 * r2 * c2 + theta3**2 *r3 * c3 + theta4**2 *r4 * c4

        # while((abs(a) > ERR_1) or (F_yield > ERR)):    # 设定迭代终止条件，即两次迭代差值不超过0.0000001
        # while(((F_yield > ERR) or (F_yield < ERR_l)) or (abs(a) > ERR_1)):

        # while(((F_yield > ERR) or (F_yield < ERR_l)) \
                # or (dp < ERR_dp) or (abs(a) > ERR_1)):

        # ********************************************
        # 当不限定a = abs(dp - dp0) < Err_1 时，似乎收敛性有很大提高~
        # ***********************************************

        sum_R_n1 = sum_R_n_trail
        R_array_n1 = R_array_n
        a_theta_n1 = a_theta_n
        c_theta_n1 = c_theta_n
        theta2_r_a_n1 = theta2_r_a_n
        theta2_r_c_n1 = theta2_r_c_n

        while(((F_yield > ERR) or (F_yield < ERR_l)) or (dp <= ERR_dp)):
        # while((F_yield > ERR) or (dp0 < ERR_dp)):
            # 在进行参数计算时不能要求F<ERR_l，否则引起牛顿迭代失效。。画图时，修正回来！！
        # while(F_yield > ERR):

            # 试应力是否需要更新!!!!*****************

            # a_theta = a1 * theta1 + a2 * theta2 + a3 * theta3
            # c_theta = theta1 * c1 + theta2 * c2 + theta3 * c3
            # theta2_r_a = theta1**2*r1*a1 + theta2**2*r2*a2 + theta3**2*r3*a3
            # theta2_r_c = theta1**2*r1*c1 + theta2**2*r2*c2 + theta3**2*r3*c3
            # sign = np.sign(stress_trial - a_theta)
            # R = (R + bios * Q * dp0)/(1 + bios * dp0)        # Q-R必须恒为正值，可考虑引入Mc bracket< >;根据迭代的dp值更新R值，准备下一次迭代求解
            # 根据dp更新R，进行下一轮迭代计算

            F = abs(stress_trial - a_theta_n1) - (sgm0 + sum_R_n1) - (3 * G + c_theta_n1) * dp0
            # differ_F = np.sign(stress_trial - a_theta ) * (theta2_r_a) - (bios1 * (Q1 - R1) + bios2 * (Q2 - R2) + \
            #             bios3) - (3 * G + c_theta) + (theta2_r_c) * dp0

            differ_R_n1 = np.sum(b_array * (Q_array - R_array_n1)) + bios3
            differ_F = np.sign(stress_trial - a_theta_n1 ) * (theta2_r_a_n1) - differ_R_n1 - (3 * G + c_theta_n1) + (theta2_r_c_n1) * dp0
            # 记录下此次迭代的屈服状态方程Fn+1的值，并判断是否满足迭代终止条件！
            F_yield = F

            # F_l.append(F_yield)
            # d_F.append(differ_F)

            # if ((F_yield < ERR) & (F_yield > ERR_l) & (abs(a) < ERR_1)):
            # if ((F_yield < ERR) & (F_yield > ERR_l) \
                # & (dp0 > ERR_dp) & (abs(a) < ERR_1)):
            if ((F_yield <= ERR) and (F_yield >= ERR_l) and (dp > ERR_dp)):
            # if ((F_yield < ERR) & (dp0 > ERR_dp)):
                # dp0 = dp0
                # dp = dp0
                break
            else:
                dp = dp0 - F / differ_F  # 更新迭代dp，准备下一次迭代求解  ## 考虑是否使用abs保证dp为正值

                # dp_l.append(dp0)

                # a = dp - dp0    ## 中间变量存储变量，控制循环终止条件！

                # dp0 = dp
                # ********************************************************
                # 在求出/更新 dp的值之后，需要将其值代入Fn+1方程中进行验证F的值，关键问题是此时其他量是否也需要更新？

                theta_array_n1 = 1 / (1 + r_array * dp)

                a_theta_n1 = np.sum(a_array_n * theta_array_n1)
                c_theta_n1 = np.sum(c_array * theta_array_n1)
                theta2_r_a_n1 = np.sum(theta_array_n1 ** 2 * r_array * a_array_n)
                theta2_r_c_n1 = np.sum(theta_array_n1 ** 2 * r_array * c_array)
                # theta1 = 1 / (1 + r1 * dp0)  # 计算、更新各个θi值
                # theta2 = 1 / (1 + r2 * dp0)
                # theta3 = 1 / (1 + r3 * dp0)
                # theta4 = 1 / (1 + r4 * dp0)

            # 试应力是否需要更新!!!!*****************

                R_array_n1 = (R_array_n + b_array * Q_array * dp) / (1 + b_array * dp)
                # R1 = (R1 + bios1 * Q1 * dp0)/(1 + bios1 * dp0)        # Q-R必须恒为正值，可考虑引入Mc bracket< >;根据迭代的dp值更新R值，准备下一次迭代求解
                # R2 = (R2 + bios2 * Q2 * dp0)/(1 + bios2 * dp0)
                R3_n1 = R3_n + bios3 * dp
                sum_R_n1 = np.sum(R_array_n1) + R3_n1
                dp0 = dp
                # print('输出dp列表：',dp_l)

                # F_yield = abs(stress_trial - a_theta) - (sgm0 + R1 + R2 + R3 + R4 + R5) - (3 * G + c_theta) * dp0
                # if ((F_yield < ERR) & (F_yield > ERR_l) & (dp0 > ERR_dp) & (abs(a) < ERR_1)):
                # if ((F_yield < ERR) & (F_yield > ERR_l) & (dp0 > ERR_dp)):
                    # dp0 = dp0
                    # break
        # print(F_yield)

        if (dp <= 0):
            print('********dp不能小于等于0!*********')
            raise ValueError

        return dp              # 返回方程的解dp，即为该加载步的等效塑性应变

if __name__ == '__main__':
    l=NewtonIteration()
    print(l)

# # from function_tools import Mc_bracket
# import numpy as np

# # def NewtonIteration(a1,a2,a3,a4,c1,c2,c3,c4,r1,r2,r3,r4,G,sgm0,Q1,R1,bios1,\
# #                     Q2,R2,bios2,R3,bios3,stress_trial):
# def NewtonIteration(ai, ci, ri, G, sgm0, Qi, Ri, R3, biosi, bios3, stress_trial):
#         """
#         通过Newton-Raphson迭代，求解该塑性加载步中产生的等效塑性应变.
#         当判断出材料在该加载步中处于塑性流动状态时，则根据上一步加载步求解结果的背应力a1，a2，a3，
#         弹性域大小R及各向同性硬化饱和值Q，以及该加载步中的试应力值，通过迭代求解dp

#         Args:   param1,2,3  a1, a2, a3    第n步加载步求解得到的 三个背应力分量
#                 param4      R             第n步加载步求解得到的 弹性域大小
#                 param5      stress_trial  第n+1步加载步的 试应力
#         """

#         dp0 = 0     #0.00000015    ## 初值的选取对计算结果有很大影响  # 假定该步等效塑性应变值的初值为0   0.00015位每个加载步的平均应变增量值

#         # dp_l = []
#         # dp_l.append(dp0)
#         # F_l = []
#         # d_F = []

#         # dp = 0.1
#         # a = dp - dp0
#         ERR = 0.000001
#         # ERR_l = -0.000001
#         ERR_l = -1
#         # ERR_1 = 0.00001
#         ERR_dp = 0
#         a_array = ai
#         R_array = Ri

#         c_array = ci
#         r_array = ri
#         b_array = biosi
#         Q_array = Qi

#         sum_a = np.sum(a_array)
#         sum_R = np.sum(R_array) + R3
#         F_yield = abs(stress_trial - sum_a) - (sgm0 + sum_R)

#         # F_l.append(F_yield)

#         # 以dp = 0作为第n+1步的初值进行迭代计算
#         # theta1 = 1 / (1 + r1 * dp0)  # 计算、更新各个θi值
#         # theta2 = 1 / (1 + r2 * dp0)
#         # theta3 = 1 / (1 + r3 * dp0)
#         # theta4 = 1 / (1 + r4 * dp0)

#         # theta_array = np.array([theta1,theta2,theta3,theta4])
#         theta_array = 1 / (1 + r_array * dp0)

#         a_theta = np.sum(a_array * theta_array)
#         c_theta = np.sum(c_array * theta_array)
#         theta2_r_a = np.sum(theta_array ** 2 * r_array * a_array)
#         theta2_r_c = np.sum(theta_array ** 2 * r_array * c_array)
#         # a_theta = a1 * theta1 + a2 * theta2 + a3 * theta3 + a4 * theta4
#         # c_theta = theta1 * c1 + theta2 * c2 + theta3 * c3 + theta4 * c4
#         # theta2_r_a = theta1**2 * r1 * a1 + theta2**2 * r2 * a2 + theta3**2 *r3 * a3 + theta4**2 *r4 * a4
#         # theta2_r_c = theta1**2 * r1 * c1 + theta2**2 * r2 * c2 + theta3**2 *r3 * c3 + theta4**2 *r4 * c4

#         # while((abs(a) > ERR_1) or (F_yield > ERR)):    # 设定迭代终止条件，即两次迭代差值不超过0.0000001
#         # while(((F_yield > ERR) or (F_yield < ERR_l)) or (abs(a) > ERR_1)):

#         # while(((F_yield > ERR) or (F_yield < ERR_l)) \
#                 # or (dp < ERR_dp) or (abs(a) > ERR_1)):

#         # ********************************************
#         # 当不限定a = abs(dp - dp0) < Err_1 时，似乎收敛性有很大提高~
#         # ***********************************************

#         while(((F_yield > ERR) or (F_yield < ERR_l)) or (dp0 < ERR_dp)):
#         # while((F_yield > ERR) or (dp0 < ERR_dp)):
#             # 在进行参数计算时不能要求F<ERR_l，否则引起牛顿迭代失效。。画图时，修正回来！！
#         # while(F_yield > ERR):

#             # 试应力是否需要更新!!!!*****************

#             # a_theta = a1 * theta1 + a2 * theta2 + a3 * theta3
#             # c_theta = theta1 * c1 + theta2 * c2 + theta3 * c3
#             # theta2_r_a = theta1**2*r1*a1 + theta2**2*r2*a2 + theta3**2*r3*a3
#             # theta2_r_c = theta1**2*r1*c1 + theta2**2*r2*c2 + theta3**2*r3*c3
#             # sign = np.sign(stress_trial - a_theta)
#             # R = (R + bios * Q * dp0)/(1 + bios * dp0)        # Q-R必须恒为正值，可考虑引入Mc bracket< >;根据迭代的dp值更新R值，准备下一次迭代求解
#             # 根据dp更新R，进行下一轮迭代计算

#             F = abs(stress_trial - a_theta) - (sgm0 + sum_R) - (3 * G + c_theta) * dp0
#             # differ_F = np.sign(stress_trial - a_theta ) * (theta2_r_a) - (bios1 * (Q1 - R1) + bios2 * (Q2 - R2) + \
#             #             bios3) - (3 * G + c_theta) + (theta2_r_c) * dp0

#             differ_R = np.sum(b_array * (Q_array - R_array)) + bios3
#             differ_F = np.sign(stress_trial - a_theta ) * (theta2_r_a) - differ_R - (3 * G + c_theta) + (theta2_r_c) * dp0
#             # 记录下此次迭代的屈服状态方程Fn+1的值，并判断是否满足迭代终止条件！
#             F_yield = F

#             # F_l.append(F_yield)
#             # d_F.append(differ_F)

#             # if ((F_yield < ERR) & (F_yield > ERR_l) & (abs(a) < ERR_1)):
#             # if ((F_yield < ERR) & (F_yield > ERR_l) \
#                 # & (dp0 > ERR_dp) & (abs(a) < ERR_1)):
#             if ((F_yield <= ERR) and (F_yield >= ERR_l) and (dp0 > ERR_dp)):
#             # if ((F_yield < ERR) & (dp0 > ERR_dp)):
#                 # dp0 = dp0
#                 break
#             else:
#                 dp0 = dp0 - F / differ_F  # 更新迭代dp，准备下一次迭代求解  ## 考虑是否使用abs保证dp为正值

#                 # dp_l.append(dp0)

#                 # a = dp - dp0    ## 中间变量存储变量，控制循环终止条件！

#                 # dp0 = dp
#                 # ********************************************************
#                 # 在求出/更新 dp的值之后，需要将其值代入Fn+1方程中进行验证F的值，关键问题是此时其他量是否也需要更新？

#                 theta_array = 1 / (1 + r_array * dp0)

#                 a_theta = np.sum(a_array * theta_array)
#                 c_theta = np.sum(c_array * theta_array)
#                 theta2_r_a = np.sum(theta_array ** 2 * r_array * a_array)
#                 theta2_r_c = np.sum(theta_array ** 2 * r_array * c_array)
#                 # theta1 = 1 / (1 + r1 * dp0)  # 计算、更新各个θi值
#                 # theta2 = 1 / (1 + r2 * dp0)
#                 # theta3 = 1 / (1 + r3 * dp0)
#                 # theta4 = 1 / (1 + r4 * dp0)

#             # 试应力是否需要更新!!!!*****************

#                 R_array = (R_array + b_array * Q_array * dp0) / (1 + b_array * dp0)
#                 # R1 = (R1 + bios1 * Q1 * dp0)/(1 + bios1 * dp0)        # Q-R必须恒为正值，可考虑引入Mc bracket< >;根据迭代的dp值更新R值，准备下一次迭代求解
#                 # R2 = (R2 + bios2 * Q2 * dp0)/(1 + bios2 * dp0)
#                 R3 = R3 + bios3 * dp0
#                 sum_R = np.sum(R_array) + R3
#                 # print('输出dp列表：',dp_l)

#                 # F_yield = abs(stress_trial - a_theta) - (sgm0 + R1 + R2 + R3 + R4 + R5) - (3 * G + c_theta) * dp0
#                 # if ((F_yield < ERR) & (F_yield > ERR_l) & (dp0 > ERR_dp) & (abs(a) < ERR_1)):
#                 # if ((F_yield < ERR) & (F_yield > ERR_l) & (dp0 > ERR_dp)):
#                     # dp0 = dp0
#                     # break
#         # print(F_yield)

#         if (dp0 < 0):
#             print('********dp不能小于等于0!*********')
#             raise ValueError

#         return dp0              # 返回方程的解dp，即为该加载步的等效塑性应变

# if __name__ == '__main__':
#     l=NewtonIteration()
#     print(l)
