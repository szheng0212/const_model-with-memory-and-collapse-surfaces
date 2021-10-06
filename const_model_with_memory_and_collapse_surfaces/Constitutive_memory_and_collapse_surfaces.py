# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 11:27:08 2021

@author: SZheng
"""

import numpy as np
# from modif_Newton_Raphson_Iteration import NewtonIteration
# from three_q_modifi_newton import NewtonIteration
from np_Newton_Raphson_Iteration import NewtonIteration
# from function_tools import  Mc_bracket
class ConstitutiveModel1D(object):

    def __init__(self, E, sgm0, c1, c2, c3, c4, cs1, cs2, cs3, cs4, h1, h2, h3, h4, bt1, bt2,\
        r1, r2, r3, r4, bios1, Q10, Q1s, bios2, Q20, Q2s, w, f0, fs, yt, fai, ro):   ## 十个参数
        """
        Args:
            param1 (array):Array containing the material parameters

        """
        # 控制两个短程背应力演化速度和短程弹性抗力演化速率相同，可以在此修改或调解该参数限制。

        # *****************
        self.E = E                    # 弹性模量 数值
        self.v = 0.3                        # 泊松比
        self.sgm0 = sgm0              # 屈服强度


        # 定义各向同性硬化参数，包括初始值，饱和值，演化速率控制参数  此处使用了两个各向同性硬化分量
        self.biosi = np.array([bios1, bios2]) # 各向同性演化速率控制参数

        self.Qi0 = np.array([Q10, Q20]) # 各向同性硬化变量R的饱和参数 的 初始值
        self.Qis = np.array([Q1s, Q2s]) # 各向同性硬化变量R的饱和参数 Q 的饱和值

        # self.Q1_n = [self.Qi0[0]]
        # self.Q2_n = [self.Qi0[1]]
        self.bti = np.array([bt1, bt2]) # 控制Q随记忆面半径演化速率的参数

        # 定义弹性模量
        self.G = self.E / (2 * (1 + self.v))     # 剪切模量

        # 定义随动硬化参数Ci的初始值，饱和值以及演化速率控制参数
        self.c0i = np.array([c1, c2, c3, c4]) # 定义参数Ci的初始值 c0i的数组
        self.csi = np.array([cs1, cs2, cs3, cs4]) # 定义参数Ci的饱和值 csi的数组
        self.hi = np.array([h1, h2, h3, h4]) # 定义控制参数Ci演化速率的参数 hi的数组
        self.ri = np.array([r1, r2, r3, r4]) # 定义动态回复项参数ri的 参数数组

        # 此处认为所有随动硬化律的动态回复项的φ函数相同
        self.f0 = f0 # 定义动态回复项系数φ(p) 函数初值 φ0 ， 一般为1
        self.fs = fs # 定义动态回复项系数φ(p) 函数的饱和值 φ∞，一般小于1
        # 定义存储 φ函数值的数组
        self.f_l_n = [f0] # 存储第n步的 φ(p)值
        # self.f_n = [self.f0]
        self.w = w # 定义函数φ(p)演化速率的函数

        # 定义记忆面模型(memory surface)的各个内变量参数信息
        self.kx_n = [0] # 表征记忆面位置的变量（类似背应力）
        self.yt = yt
        self.fai = fai
        self.ro = ro


        self.q_n = [0]       # 存储第n步的记忆面 半径
        self.q_max = [0]     # 存储 整个加载历史中的 最大记忆面半径



        # 定义消散目标面模型(collapse target surface)的各个内变量参数信息

        self.peps_un = [0]
        self.e_un = [0]      # 存储 每一个卸载步 所对应的 塑性应变空间中
                            # 塑性应变与记忆面位置张量之间的差值
        self.q_u_n = [0]      # 重建目标面(CTS)在第n步的实时更新的距离上一次卸载点之间的
                            # 塑性应变范数距离


        # 定义背应力分量的存储数组
        self.a1 = [0]                  # 记录每个加载步的背应力分量a1
        self.a2 = [0]                  # 记录每个加载步的背应力分量a2
        self.a3 = [0]                  # 记录每个加载步的背应力分量a3
        self.a4 = [0]
        self.a = [0]                   # 记录每个加载步的背应力a

        self.da1 = []
        self.da2 = []
        self.da3 = []
        self.da4 = []
        self.da = []

        self.R1 = [0]
        self.dR1 = []                   # 记录每个加载步的弹性域大小R
        self.dR2 = []                   # 记录每个加载步的弹性域大小增量 dR
        self.R2 = [0]                   # 记录每个加载步的弹性域大小R
        self.dR = []
        self.R = [0]

        self.p = [0]                   # 记录每个加载步的累积塑性应变p
        self.dp = []                  # 记录每个加载步的累积塑性应变增量dp

        # 第一次计算即调用了dpeps的值，故给定初值为0，在计算过程结束后可删除首元0
        self.dpeps = [0]               # 记录每个加载步的塑性应变增量 dpeps

        self.deeps = []
        self.sgm = [0]                 # 记录每个加载步的应力 sgm
        self.peps = [0]                # 记录每个加载步的塑性应变值peps
        self.eeps = [0]


        # 定义各向同性硬化 内变量的信息值
        self.Q1_n = [self.Qi0[0]] # 第n步的各向同性硬化变量R的饱和值，与记忆面半径相关
        self.Q2_n = [self.Qi0[1]]


        # 定义 储存 随动硬化律 参数变量的数组
        self.c1_n = [self.c0i[0]]   # 存储每一步计算的 随动硬化参数Ci的值 ， 随塑性发展不断演化
        self.c2_n = [self.c0i[1]]
        self.c3_n = [self.c0i[2]]
        self.c4_n = [self.c0i[3]]







    def step_solve(self, teps_n1, teps_n):
        ## 该模块根据当前状态完成一个增量步的计算并更新信息

        '''
        Args:
            param1: ChabocheModelID object ChabocheModel1D 类对象 以类对象为参数不好，容易造成内存过大~~~~~数据传输缓慢
            param2: the total strain value (n+1th step)  teps 第n+1步的总应变
        '''
        ### 将上一步（第n步计算的状态变量结果提取出来，以便进行新的迭代）

        # 读取 第n步 的背应力分量内变量的值
        ai_n = np.array([self.a1[-1], self.a2[-1], self.a3[-1] ,self.a4[-1]])
        a_n = self.a[-1]

        # 读取第n步 的随动硬化参数Ci的值
        c1_n = self.c1_n[-1]
        c2_n = self.c2_n[-1]
        c3_n = self.c3_n[-1]
        c4_n = self.c4_n[-1]

        # 读取第n步 的φ函数的值
        f_n = self.f_l_n[-1]

        # 读取 第n步 的记忆面模型的 内变量的相关信息值
        kx_n = self.kx_n[-1]
        q_n = self.q_n[-1]
        q_max_n = self.q_max[-1]



        # 读取第n步 的崩溃目标面模型参数数据(第n步计算结果可以用来判定 下一步的试计算状态是扩张还是崩溃效应被激活)
        q_u_n = self.q_u_n[-1]
        e_un = self.e_un[-1] # 读取上一次卸载时的 塑性应变相对于 记忆面中心 的位置
        peps_un = self.peps_un[-1]


        # 读取 第n步 的各向同性硬化变量信息值

        Ri_n = np.array([self.R1[-1], self.R2[-1]]) # 读取第n步的各个 弹性域半径值
        R_n = self.R[-1]    # 读取第n步的总弹性域半径值
        Q1_n = self.Q1_n[-1]  # 读取第n步的 R1的饱和值 Q1数值
        Q2_n = self.Q2_n[-1]  # 读取第n步的 R2的饱和值 Q2数值


        # ******************************************
        p_n = self.p[-1]    # 读取第n步的累积塑性应变值,用以计算φ值

        # 读取第n步的塑性应变,用以判断第n步是否在记忆面上,以及是否 试计算的状态有
        # 沿着崩溃目标面外法线运动的趋势，用以判断记忆面和崩溃目标面的演化方式
        peps_n = self.peps[-1]
                                #
        # dpeps_n = self.dpeps[-1]    # 读取第n步的塑性应变增量
        sgm_n = self.sgm[-1]        # 读取第n步的应力

        eeps_n = self.eeps[-1]  #读取第n步的弹性应变


        #  计算试应力
        stress_trial = sgm_n + self.E * (teps_n1 - teps_n)

        # 根据第n+1步加载应变计算试应力！

        # stress_trial = sgm + self.E * (teps_n1 - teps_n)
        # E_eff = self.E * (1 - self.E / (self.E + self.c1 + self.c2 +\
        #                 self.c3 - self.r1 * a1 - self.r2 * a2 - self.r3 * a3 +\
        #                     self.bios * (self.Q - R)))
        # stress_trial = sgm + E_eff * (teps_n1-teps_n)
        # stress_trial = self.E * (teps_n1 - teps_n + eeps)                   # 计算第 n+1步 试应力 n+1步总应变-n步塑性应变

        ERR = 0
        # F_yield = abs(stress_trial - a1 - a2 - a3 - a4) - self.sgm0 -\
        #     (R1 + R2 + R3)
        # F_yield = abs(stress_trial - a) - self.sgm0 - R
        F_yield = abs(stress_trial - a_n) - (self.sgm0 + R_n) # 计算第n+1步的 试状态 是否仍旧满足屈服方程，满足则接受为弹性状态


        if (F_yield <= ERR): # 判断为弹性状态

            # 需要看是否为卸载状态，若为第一步的卸载状态，则保存上一步的塑性应变进而计算卸载张量 e_un 和 q_un
            #if (dpeps_n != 0): # 当前步（第n+1步）判据为弹性，而第n步（上一计算步）为塑性，则判定该计算步为第一卸载步，
                                        # 则上一计算步即为卸载点，保留其记忆面模型的内变量信息值
            #   e_un_n1 = peps_n - kx_n
            #  self.e_un.append(e_un_n1)

            # 判定此加载步为弹性计算，则保持该计算步内的塑性应变增量为0，背应力不变，背应力系数不变，记忆面 半径和位置不变
            sgm_n1 = stress_trial
            deeps_n1 = teps_n1 - teps_n    # 弹性应变增量
            eeps_n1 = eeps_n + deeps_n1

            self.dp.append(0)
            self.p.append(p_n)
            self.dpeps.append(0)
            self.peps.append(peps_n)

            self.deeps.append(deeps_n1)
            self.eeps.append(eeps_n1)

            self.sgm.append(sgm_n1)


            # 更新随动硬化参数、内变量等值(保持不变,添加最末位数据)
            self.c1_n.append(c1_n)   # 存储每一步计算的 随动硬化参数Ci的值 ， 随塑性发展不断演化
            self.c2_n.append(c2_n)
            self.c3_n.append(c3_n)
            self.c4_n.append(c4_n)

            # 更新φ(p)函数值
            self.f_l_n.append(f_n)

            # 更新各向同性硬化律相关内变量值的数组
            self.Q1.append(Q1_n)
            self.Q2.append(Q2_n)

            self.R1.append(Ri_n[0])
            self.R2.append(Ri_n[1])
            self.R.append(R_n)

            self.dR1.append(0)
            self.dR2.append(0)
            self.dR.append(0)


            # 更新记忆面模型相关参数
            self.kx_n.append(kx_n) # 表征记忆面位置的变量（类似背应力）
            self.q_n.append(q_n)    # 存储第n步的记忆面 半径
            self.q_max.append(q_max_n)     # 存储 整个加载历史中的 最大记忆面半径

            # 定义消散目标面模型(collapse target surface)的各个内变量参数信息
            self.peps_un.append(peps_un)
            # 存储 每一个卸载步 所对应的 塑性应变空间中,塑性应变与记忆面位置张量之间的差值
            self.e_un.append(e_un)

            self.q_u_n.append(q_u_n)      # 重建目标面(CTS)在第n步的实时更新的距离上一次卸载点之间的
                                # 塑性应变范数距离
                # 更新背应力分量数组
            self.a1.append(ai_n[0])
            self.a2.append(ai_n[1])
            self.a3.append(ai_n[2])
            self.a4.append(ai_n[3])
            self.a.append(a_n)
                # 更新背应力分量增量数组
            self.da1.append(0)
            self.da2.append(0)
            self.da3.append(0)
            self.da4.append(0)
            self.da.append(0)


        else:        # 判定该加载步进入塑性状态
            # 判断此计算步为继续加载还是反向加载
            # 首先假定该计算步的等效塑性应变初值为0,p=0, 通过牛顿迭代计算出一个 试等效塑性应变和试塑性应变,
            # 通过结合第n步的计算塑性状态,来

                    # 通过Newton-Raphson迭代求解第 n+1步 产生的等效塑性应变dp
            dpn1 = NewtonIteration(ai_n,self.ci,self.ri,self.G,self.sgm0,self.Qi,Ri_n,R3_n,self.biosi,self.bios3,stress_trial)      # 调用dp求解函数迭代求解该加载步的dp
                    # 由dp值，分别更新 dpeps，peps，da1，da2，da3，a1，a2，a3，a，dR，R，sgm
            # 更新θn+1
            theta_n1 = 1 / (1 + self.ri * dpn1)
            dRi_n1 = (self.biosi * (self.Qi - Ri_n) * dpn1) / (1 + self.biosi * dpn1)
            dR3_n1 = (self.bios3 * dpn1)
            dR_n1 = np.sum(dRi_n1) + dR3_n1
            Ri_n1 = (Ri_n + self.biosi * self.Qi * dpn1) / (1 + self.biosi * dpn1)
            R3_n1 = (R3_n + self.bios3 * dpn1)
            R_n1 = np.sum(Ri_n1) + R3_n1
            # theta1 = 1 / (1 + self.r1 * dpn1)
            # theta2 = 1 / (1 + self.r2 * dpn1)
            # theta3 = 1 / (1 + self.r3 * dpn1)
            # theta4 = 1 / (1 + self.r4 * dpn1)
            # theta = np.array([theta1, theta2, theta3, theta4])
            # 更新Rn+1
            # Ri = np.array([R1, R2])
            # dRi = (self.biosi * (self.Qi - Ri) * dpn1) / (1 + self.biosi * dpn1)
            # dR3 = (self.bios3 * dpn1)
            # dR1 = (self.bios1 * (self.Q1 - R1) * dpn1) / (1 + self.bios1 * dpn1)
            # dR2 = (self.bios2 * (self.Q2 - R2) * dpn1) / (1 + self.bios2 * dpn1)
            # dR3 = (self.bios3 * dpn1) #/ (1 + self.bios5 * dpn1)
            # dR = np.sum(dRi) + dR3

            # Ri = (Ri + self.biosi * self.Qi * dpn1) / (1 + self.biosi * dpn1)
            # R1 = (R1 + self.bios1 * self.Q1 * dpn1) / (1 + self.bios1 * dpn1)
            # R2 = (R2 + self.bios2 * self.Q2 * dpn1) / (1 + self.bios2 * dpn1)
            # R3 = (R3 + self.bios3 * dpn1) #/ (1 + self.bios1 * dpn1)
            # R = np.sum(Ri) + R3

            # 计算Sn+1 - αn+1
            # stress_trial_n1 = stress_trial - (theta1 * a1 + theta2 * a2 + theta3 * a3 + theta4 * a4)
            # stress_trial_n1 = stress_trial - (np.sum(theta * ai))
            # S_a = (self.sgm0 + R) * abs(stress_trial_n1) * np.sign(stress_trial_n1) / ((self.sgm0 + R) + \
            #         (3 * self.G + np.sum(theta * self.ci)) * dpn1)
            # n_direction = np.sign(S_a)
            stress_trial_n1 = stress_trial - (np.sum(theta_n1 * ai_n))
            S_a = (self.sgm0 + R_n1) * abs(stress_trial_n1) * np.sign(stress_trial_n1) / ((self.sgm0 + R_n1) + \
                    (3 * self.G + np.sum(theta_n1 * self.ci)) * dpn1)
            n_direction_n1 = np.sign(S_a)

            # 依次更新Sn+1 - αn+1,塑性流动方向nn+1,塑性应变增量,背应力,应力增量**********************

            dpeps_n1 = n_direction_n1 * dpn1  ## 计算这一步的塑性应变增量
            peps_n1 = peps_n + dpeps_n1     ## 更新本步的总塑性应变 n+1步的值
            deeps_n1 = teps_n1 - teps_n - dpeps_n1
            eeps_n1 = eeps_n + deeps_n1
            # stress_trial_n1 = stress_trial - (theta1*a1 + theta2*a2 + theta3*a3 + theta4*a4)
            # dpeps = n_direction * dpn1  ## 计算这一步的塑性应变增量
            # peps = peps + dpeps     ## 更新本步的总塑性应变 n+1步的值
            # deeps = teps_n1 - teps_n - dpeps
            # eeps = eeps + deeps
            # da1 = (self.c1*dpeps - self.r1*a1*dp)/(1 + self.r1 * dp)             # 计算背应力分量增量
            # da2 = (self.c2*dpeps - self.r2*a2*dp)/(1 + self.r2 * dp)
            # da3 = (self.c3*dpeps - self.r3*a3*dp)/(1 + self.r3 * dp)
            # da = da1 + da2 + da3

            # a1 = (a1 + self.c1*dpeps)/(1 + self.r1 * dp)
            # a2 = (a2 + self.c2*dpeps)/(1 + self.r2 * dp)
            # a3 = (a3 + self.c3*dpeps)/(1 + self.r3 * dp)
            # a = a1 +a2 + a3

            # dR = (self.bios * (self.Q-R) * dp)/(1+self.bios * dp)
            # R = (R + self.bios * self.Q * dp)/(1 + self.bios * dp)
            dai_n1 = (self.ci * dpeps_n1 - self.ri * ai_n * dpn1) / (1 + self.ri * dpn1)
            # da1 = (self.c1*dpeps - self.r1*a1*dpn1) / (1 + self.r1 * dpn1)             # 计算背应力分量增量
            # da2 = (self.c2*dpeps - self.r2*a2*dpn1) / (1 + self.r2 * dpn1)
            # da3 = (self.c3*dpeps - self.r3*a3*dpn1) / (1 + self.r3 * dpn1)
            # da4 = (self.c4*dpeps - self.r4*a4*dpn1) / (1 + self.r4 * dpn1)
            da_n1 = np.sum(dai_n1)

            ai_n1 = (ai_n + self.ci * dpeps_n1) / (1 + self.ri * dpn1)
            # a1 = (a1 + self.c1*dpeps) / (1 + self.r1 * dpn1)
            # a2 = (a2 + self.c2*dpeps) / (1 + self.r2 * dpn1)
            # a3 = (a3 + self.c3*dpeps) / (1 + self.r3 * dpn1)
            # a4 = (a4 + self.c4*dpeps) / (1 + self.r4 * dpn1)
            a_n1 = np.sum(ai_n1)

            # dR = (self.bios * Mc_bracket(self.Q-R) * dp)/(1+self.bios * dp)
            # dR = (self.bios * (self.Q-R) * dp)/(1+self.bios * dp)
            # dR1 = (self.bios1 * (self.Q1 - R1) * dpn1) / (1 + self.bios1 * dpn1)
            # dR2 = (self.bios2 * (self.Q2 - R2) * dpn1) / (1 + self.bios2 * dpn1)
            # dR3 = (self.bios3 * dpn1) #/ (1 + self.bios5 * dpn1)
            # dR = dR1 + dR2 + dR3

            # R1 = (R1 + self.bios1 * self.Q1 * dpn1) / (1 + self.bios1 * dpn1)
            # R2 = (R2 + self.bios2 * self.Q2 * dpn1) / (1 + self.bios2 * dpn1)
            # R3 = (R3 + self.bios3 * dpn1) #/ (1 + self.bios1 * dpn1)
            # R = R1 + R2 + R3
            # R = R + dR
            # p = p + dpn1
            p_n1 = p_n + dpn1


            # sgm = stress_trial - 2 * self.G * dpeps # 根据第n+1步的弹性应变更新应力，而不是直接按照总的计算，好像可以减少步计算的误差???
            # sgm = sgm + self.E * deeps
            # sgm = stress_trial - 2 * self.G * dpeps
            # sgm = sgm + E_eff * (teps_n1 - teps_n)
            # sgm_n1 = stress_trial - 2 * self.G * dpeps_n1
            # sgm_n1 = sgm_n + self.E * (teps_n1 - teps_n) - 2 * self.G * dpeps_n1
            # sgm_n1 = stress_trial - 3 * self.G * dpeps_n1
            sgm_n1 = sgm_n + self.E * deeps_n1
            # sgm = sgm + E_eff * (teps_n1 - teps_n)

            self.a1.append(ai_n1[0])
            self.a2.append(ai_n1[1])
            self.a3.append(ai_n1[2])
            self.a4.append(ai_n1[3])
            self.a.append(a_n1)


            self.da1.append(dai_n1[0])
            self.da2.append(dai_n1[1])
            self.da3.append(dai_n1[2])
            self.da4.append(dai_n1[3])
            self.da.append(da_n1)

            self.R1.append(Ri_n1[0])
            self.R2.append(Ri_n1[1])
            self.R3.append(R3_n1)
            self.R.append(R_n1)

            self.dR1.append(dRi_n1[0])
            self.dR2.append(dRi_n1[1])
            self.dR3.append(dR3_n1)
            self.dR.append(dR_n1)

            self.dp.append(dpn1)
            # self.p.append(dpn1)
            self.p.append(p_n1)
            self.dpeps.append(dpeps_n1)
            self.peps.append(peps_n1)
            self.deeps.append(deeps_n1)
            self.eeps.append(eeps_n1)
            self.sgm.append(sgm_n1)



    def total_solve(self, l_strain):       ## l_strain:list[],即接受一个加载的试验应变列表，通过迭代循环求解每一步的计算结果
        # 列表类型是可变的，接受列表参数可对列表直接进行修改，不需要返回值。

        l_strain = np.array(l_strain)

        self.step_solve(l_strain[0], 0)

        # for i in range(1, len(l_strain)):
        for strain in range(1, len(l_strain)):
            self.step_solve(l_strain[strain], l_strain[strain - 1])    ## 直接修改了该类ChabocheModel对象的self属性值！
        ## 删除列表初始化造成的无用初值 0
        del self.sgm[0]

        del self.a1[0]
        del self.a2[0]
        del self.a3[0]
        del self.a4[0]
        del self.a[0]

        del self.R1[0]
        del self.R2[0]
        del self.R3[0]
        del self.R[0]

        # del self.dR1[0]
        # del self.dR2[0]
        # del self.dR3[0]
        # del self.dR[0]

        del self.peps[0]
        del self.p[0]
        del self.eeps[0]

        # return self.sgm,self.da1,self.da2,self.da3,self.da4,self.da,\
        #     self.a1,self.a2,self.a3,self.a4,self.a,\
        #         self.dR1,self.dR2,self.dR3,self.dR,\
        #             self.R1,self.R2,self.R3,self.R,\
        #                 self.dpeps,self.peps,self.dp,self.p
        return self.sgm

        # return self.da1
        # return self.da2
        # return self.da3
        # return self.da4
        # return self.da
        # return self.a1
        # return self.a2
        # return self.a3
        # return self.a4
        # return self.a

        # return self.dR1
        # return self.dR2
        # return self.dR3
        # return self.dR
        # return self.R1
        # return self.R2
        # return self.R3
        # return self.R

        # return self.dpeps
        # return self.peps
        # return self.deeps
        # return self.eeps
        # return self.dp
        # return slef.p

if __name__ == '__main__':
    from function_tools import str_to_float
    import xlrd
    import matplotlib.pyplot as plt
    from modif_goal_fuction import goal_function

    # workbook = xlrd.open_workbook(r'C:\\Users\\SZheng\\Desktop\\HA6-1.xlsx')                     # 打开文件
    workbook = xlrd.open_workbook(r'F:\\SZheng\\material_test\\data_manipulate\\Output_HA_6_2_1-30_4.xlsx')
    ### wb = Workbook()
    ### wb = load_workbook("temp.xlsx")
    ##xlsx_path = ''                                            # 手动设置试验数据文件路径
    #   (r'C:\\Users\\SZheng\\Desktop\\zzz.xlsx')                     # 打开文件
    table = workbook.sheet_by_name('Sheet1')
    strain = table.col_values(0)              # 读取数据列 应变 列表list
    stress = table.col_values(1)              # 读取数据列 应力 列表list
    str_to_float(strain)
    str_to_float(stress)

    l = [166210,215.79,15433.65,10072.61,251.41,990.72,993.92,0.0061,0.2586,849.79]

    testmodel = ChabocheModel1D(*l)
    theory_stress = testmodel.total_solve(strain)
    print (theory_stress)
    origin_goal_value,weight_goal_value = goal_function(theory_stress,stress,strain)
    print(origin_goal_value,weight_goal_value)

    plt.figure()
    plt.plot(strain,theory_stress)
    plt.show()
    plt.figure()
    plt.plot(strain,stress)
    plt.show()
    plt.figure()
    plt.plot(strain,theory_stress)
    plt.plot(strain,stress)
    plt.show()
