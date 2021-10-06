# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 00:08:46 2021

@author: SZheng
"""

# from function_tools import Mc_bracket
import numpy as np
from numpy.lib import index_tricks
from numpy.lib.histograms import histogram

from Heaviside import Heaviside
from Steffensen_Iteration import Steffensen

# def NewtonIteration(a1,a2,a3,a4,c1,c2,c3,c4,r1,r2,r3,r4,G,sgm0,Q1,R1,bios1,\
#                     Q2,R2,bios2,R3,bios3,stress_trial):
def NewtonIteration(ai_n, ci_n, csi, ri, G, sgm0, Qi_n, Qis, Ri_n, biosi, bti, hi, stress_trial, f_n, fs, w, q_n, kx_n, peps_n, q_u_n, peps_un):
        """
        通过Newton-Raphson迭代，求解该塑性加载步中产生的等效塑性应变.
        当判断出材料在该加载步中处于塑性流动状态时，则根据上一步加载步求解结果的背应力a1，a2，a3，
        弹性域大小R及各向同性硬化饱和值Q，以及该加载步中的试应力值，通过迭代求解dp

        Args:   param1,2,3  a1, a2, a3    第n步加载步求解得到的 三个背应力分量
                param4      R             第n步加载步求解得到的 弹性域大小
                param5      stress_trial  第n+1步加载步的 试应力
        上述参数ai，ci，ri，Qi，Ri，biosi均为numpy数组
        """
        a_array_n = np.array(ai_n)
        R_array_n = np.array(Ri_n)

        c_array_n = np.array(ci_n)
        r_array = np.array(ri)
        # 控制R演化速率的参数
        b_array = np.array(biosi)
        # 控制Q演化速率的参数
        bt_array = np.array(bti)
        cs_array = np.array(csi)
        h_array = np.array(hi)

        Q_array_n = np.array(Qi_n) # 第n计算步的R的饱和值 Q, 是一个随记忆面半径变化而变化的内变量
        Qi_s_array = np.array(Qis)
        sum_R_n = np.sum(R_array_n)
        # sum_a_n_trial = np.sum(a_array_n)
        # sum_R_n_trail = np.sum(R_array_n)
        """
        当p=0初值时，第一轮无论如何使用steffensen迭代都不会改变p=0，因此，在第一轮计算p时，直接计算θn+1的
        初始值，而后用该值进行牛顿迭代求出一个非零的p值，再按照史蒂芬森迭代求出第k个p下的第k+1个θn+1值，即认为
        是该pk值下的θn+1收敛值。（在此认为θ不是p的函数，在进行牛顿迭代求解时，不再对θ求导）
        """
        # 求p=0下的θn+1值
        dp_n1 = 0
        dp_n10 = 0
        # 计算φ函数的值
        f_n1 = fs + (f_n - fs) * np.exp(-w * dp_n1)
        # 根据dp值和q_n1值 计算a_n1的值和R_n1的值,在第一次计算中,dp=0,则与第n步的参数值相同
        # 计算dp=0下的q_n1,Q_n1,R_n1
        # dq_n10= 0
        dq_n1 = 0
        Q_array_n1 = (Q_array_n + bt_array * Qi_s_array * dq_n1) / (1 + bt_array * dq_n1)
        R_array_n1 = (R_array_n + b_array * Q_array_n1 * dp_n10) / (1 + b_array * dp_n10)
        # 求R1_n1的和，即为各向同性硬化分量之和
        sum_R_n1 = np.sum(R_array_n1)
        # 计算dp=0下的C_n1
        c_array_n1 = (c_array_n + h_array * dq_n1 * cs_array) / (1 + bt_array * dq_n1)

        # 在dp_n1=0的初值条件下，R_n1=R_n，coefficient_0=coefficient,故而此处仅进行区分，计算意义不大
        # coefficient_0 = np.array(1 + f_n * r_array * dp_n10)
        coefficient = np.array(1 + f_n1 * r_array * dp_n10)

        theta_n1 = 1 / (sgm0 + sum_R_n1 + np.sum((c_array_n1 * dp_n10) / coefficient))
        # 利用该值进行第一轮Newton迭代,而后进行Steffensen迭代

        sign = stress_trial - np.sum(a_array_n / coefficient)
        F_yield = abs(sign) - (1 + 3 * G * dp_n10 * theta_n1) / theta_n1
        """
        上式即为当dp=0时的单轴形式的第n+1步的试计算屈服方程表达式，当F_yield值小于0时即认为该计算步
        为弹性状态；反之，则为塑性状态，进行迭代计算dp_n1
        """
        ERR = 0.000001
        ERR_l = -1
        ERR_dp_n1 = 0

        while(((F_yield > ERR) or (F_yield < ERR_l)) or (dp_n1 <= ERR_dp_n1)):
            F_uniaxial = abs(sign) - (1 + 3 * G * dp_n10 * theta_n1) / theta_n1
            differ_f_n1 = w * (fs - f_n) * np.exp(-w * dp_n10)
            differ_F_uniaxial = np.sign(sign) * np.sum((differ_f_n1 * r_array * dp_n10 + f_n1 * r_array) * a_array_n / (coefficient ^ 2)) - 3 * G
            F_yield = F_uniaxial
            # 至此进行了一轮牛顿迭代所需参数的计算，然后判断是否满足终止条件并继续迭代求解。
            if ((F_yield <= ERR) and (F_yield >= ERR_l) and (dp_n1 > ERR_dp_n1)):
                break
            else:
                dp_n1 = dp_n10 - F_uniaxial / differ_F_uniaxial
                """
                根据上式计算得到的 迭代等效塑性应变增量，计算该计算步该迭代步内的塑性应变增量，以及第n+1步的塑性应变，
                并根据该计算结果状态判断记忆面和消散面的相关演化方式
                """
                # 计算该dp_n1值下的θ_n1的初值,此处以θ_n1_0表示
                # 计算θ_n1_0时,是否每一步的R_n1都需要更新?

                # 需要计算该最新的dp_n1值下的f_n1值,以便代入计算
                f_n1 = f_n + abs(dp_n1)

                # 计算对应的coefficient值,该值在以该dp_n1值下的整个计算过程中保持不变
                coefficient = np.array(1 + f_n1 * r_array * dp_n1)
                theta_n1_0 = 1 / (sgm0 + sum_R_n + np.sum((c_array_n * dp_n10) / coefficient))
                # 利用上式的theta_n1_0初值进行隐式依赖链式关系迭代，求出三个值后进行steffensen固定点迭代计算
                theta_n1_1 = Steffensen()


                # 计算在经过迭代后 更新的 dp_n1值 来更新该试算步和试算状态下的 塑性流动方向
                coefficient = np.array(1 + f_n1 * r_array * dp_n1)
                direction_n1_trial = np.sign((theta_n1 / (1 + 3 * G * dp_n1 * theta_n1)) * (stress_trial - np.sum(a_array_n / coefficient)))
                # 更新试状态 的塑性应变增量和总塑性应变
                dpeps = direction_n1_trial * dp_n1
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

                if index_c_n1_trial > 0:
                    # 此时 试塑性应变 有向崩溃面 外部运动 或在超球面上 的趋势,不激活崩溃效应,对应的是index_c = 1或2
                    # 使用相应的 扩张的演化公式进行计算
                    if index_g_n1_trial > 0:
                        # 在记忆面外，记忆面 发生扩张且移动，且崩溃面需要实时更新
                        gama_dp_n1 = 

                
                
                
                
                
                




                elif index_c_n1_trial == 0:
                    # 说明此试计算得到的试塑性应变有 向崩溃面内部运动的趋势对应的是index_c = 0







