# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 19:52:29 2021

@author: SZheng
"""

# from function_tools import Mc_bracket
import numpy as np

def NewtonIteration(a1,a2,a3,a4,a5,c1,c2,c3,c4,c5,r1,r2,r3,r4,G,sgm0,Q1,R1,bios1,\
                    Q2,R2,bios2,Q3,R3,bios3,Q4,R4,bios4,R5,bios5,stress_trial):

        """
        通过Newton-Raphson迭代，求解该塑性加载步中产生的等效塑性应变.
        当判断出材料在该加载步中处于塑性流动状态时，则根据上一步加载步求解结果的背应力a1，a2，a3，
        弹性域大小R及各向同性硬化饱和值Q，以及该加载步中的试应力值，通过迭代求解dp

        Args:   param1,2,3  a1, a2, a3    第n步加载步求解得到的 三个背应力分量
                param4      R             第n步加载步求解得到的 弹性域大小
                param5      stress_trial  第n+1步加载步的 试应力
        """

        dp0 = 0.0001     #0.00000015    ## 初值的选取对计算结果有很大影响  # 假定该步等效塑性应变值的初值为0   0.00015位每个加载步的平均应变增量值
        dp = 0.1
        # a = dp - dp0
        ERR = 0.000001
        ERR_l = -1
        # ERR_1 = 0.00001
        ERR_dp = 0
        F_yield = abs(stress_trial - (a1 + a2 + a3 + a4 + a5)) - (sgm0 + R1 + R2 + R3 + R4 + R5)

        # 以dp = 0作为第n+1步的初值进行迭代计算
        theta1 = 1 / (1 + r1 * dp0)  # 计算、更新各个θi值
        theta2 = 1 / (1 + r2 * dp0)
        theta3 = 1 / (1 + r3 * dp0)
        theta4 = 1 / (1 + r4 * dp0)
        theta5 = 1

        a_theta = a1 * theta1 + a2 * theta2 + a3 * theta3 + a4 * theta4 + a5 * theta5
        c_theta = theta1 * c1 + theta2 * c2 + theta3 * c3 + theta4 * c4 + theta5 * c5
        theta2_r_a = theta1**2 * r1 * a1 + theta2**2 * r2 * a2 + theta3**2 *r3 * a3 + theta4**2 * r4 * a4 #+ theta5**2 * r5 * a5
        theta2_r_c = theta1**2 * r1 * c1 + theta2**2 * r2 * c2 + theta3**2 *r3 * c3 + theta4**2 * r4 * c4 #+ theta5**2 * r5 * c5

        # while((abs(a) > ERR_1) or (F_yield > ERR)):    # 设定迭代终止条件，即两次迭代差值不超过0.0000001
        # while(((F_yield > ERR) or (F_yield < ERR_l)) or (abs(a) > ERR_1)):

        # while(((F_yield > ERR) or (F_yield < ERR_l)) \
                # or (dp < ERR_dp) or (abs(a) > ERR_1)):

        # ********************************************
        # 当不限定a = abs(dp - dp0) < Err_1 时，似乎收敛性有很大提高~
        # ***********************************************

        while(((F_yield > ERR) or (F_yield < ERR_l)) or (dp < ERR_dp)):
        # while(F_yield > ERR):

            # 试应力是否需要更新!!!!*****************

            # a_theta = a1 * theta1 + a2 * theta2 + a3 * theta3
            # c_theta = theta1 * c1 + theta2 * c2 + theta3 * c3
            # theta2_r_a = theta1**2*r1*a1 + theta2**2*r2*a2 + theta3**2*r3*a3
            # theta2_r_c = theta1**2*r1*c1 + theta2**2*r2*c2 + theta3**2*r3*c3
            # sign = np.sign(stress_trial - a_theta)
            # R = (R + bios * Q * dp0)/(1 + bios * dp0)        # Q-R必须恒为正值，可考虑引入Mc bracket< >;根据迭代的dp值更新R值，准备下一次迭代求解
            # 根据dp更新R，进行下一轮迭代计算

            F = abs(stress_trial - a_theta) - (sgm0 + R1 + R2 + R3 + R4 + R5) - (3 * G + c_theta) * dp0
            differ_F = np.sign(a_theta - stress_trial) * (theta2_r_a) - (bios1 * (Q1 - R1) + bios2 * (Q2 - R2) + \
                        bios3 * (Q3 - R3) + bios4 * (Q4 - R4) + bios5) - (3 * G + c_theta) + (theta2_r_c) * dp0

            # 记录下此次迭代的屈服状态方程Fn+1的值，并判断是否满足迭代终止条件！
            F_yield = F

            # if ((F_yield < ERR) & (F_yield > ERR_l) & (abs(a) < ERR_1)):
            # if ((F_yield < ERR) & (F_yield > ERR_l) \
                # & (dp0 > ERR_dp) & (abs(a) < ERR_1)):
            if ((F_yield < ERR) and (F_yield > ERR_l) and (dp0 > ERR_dp)):
                dp = dp0
                break
            else:
                dp = dp0 - F / differ_F  # 更新迭代dp，准备下一次迭代求解  ## 考虑是否使用abs保证dp为正值
                # a = dp - dp0    ## 中间变量存储变量，控制循环终止条件！

                dp0 = dp
                # ********************************************************
                # 在求出/更新 dp的值之后，需要将其值代入Fn+1方程中进行验证F的值，关键问题是此时其他量是否也需要更新？
                # F_yield = abs(stress_trial - a_theta) - (sgm0 + R1 + R2 + R3 + R4 + R5) - (3 * G + c_theta) * dp0
                # if ((F_yield < ERR) & (F_yield > ERR_l) & (dp0 > ERR_dp)):
                    # dp = dp0
                    # break
                #  *****************************************************************
                if ((r1 * dp0 == -1) or (r2 * dp0 == -1) or (r3 * dp0 == -1) or (r4 * dp0 == -1)):
                    print(dp0)
                    print(F_yield,differ_F)
                    raise ValueError
                theta1 = 1 / (1 + r1 * dp0)  # 计算、更新各个θi值
                theta2 = 1 / (1 + r2 * dp0)
                theta3 = 1 / (1 + r3 * dp0)
                theta4 = 1 / (1 + r4 * dp0)
                theta5 = 1      #/ (1 + r5 * dp0)

            # 试应力是否需要更新!!!!*****************
                a_theta = a1 * theta1 + a2 * theta2 + a3 * theta3 + a4 * theta4 + a5 * theta5
                c_theta = theta1 * c1 + theta2 * c2 + theta3 * c3 + theta4 * c4 + theta5 * c5
                theta2_r_a = theta1**2 * r1 * a1 + theta2**2 * r2 * a2 + theta3**2 *r3 * a3 + theta4**2 * r4 * a4 #+ theta5**2 * r5 * a5
                theta2_r_c = theta1**2 * r1 * c1 + theta2**2 * r2 * c2 + theta3**2 *r3 * c3 + theta4**2 * r4 * c4 #+ theta5**2 * r5 * c5

                R1 = (R1 + bios1 * Q1 * dp0)/(1 + bios1 * dp0)        # Q-R必须恒为正值，可考虑引入Mc bracket< >;根据迭代的dp值更新R值，准备下一次迭代求解
                R2 = (R2 + bios2 * Q2 * dp0)/(1 + bios2 * dp0)
                R3 = (R3 + bios3 * Q3 * dp0)/(1 + bios3 * dp0)
                R4 = (R4 + bios4 * Q4 * dp0)/(1 + bios4 * dp0)
                R5 = R5 + bios5 * dp0
                # print('输出dp列表：',dp_l)

                # F_yield = abs(stress_trial - a_theta) - (sgm0 + R1 + R2 + R3 + R4 + R5) - (3 * G + c_theta) * dp0
                # if ((F_yield < ERR) & (F_yield > ERR_l) & (dp0 > ERR_dp) & (abs(a) < ERR_1)):
                # if ((F_yield < ERR) & (F_yield > ERR_l) & (dp0 > ERR_dp)):
                    # dp = dp0
                    # break
        # print(F_yield)

        if (dp < 0):
            print('********dp不能小于等于0!*********')
            raise ValueError

        return dp              # 返回方程的解dp，即为该加载步的等效塑性应变

if __name__ == '__main__':
    l=NewtonIteration()
    print(l)
