# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 21:14:44 2021

@author: SZheng
"""

import numpy as np
# from modif_Newton_Raphson_Iteration import NewtonIteration
# from three_q_modifi_newton import NewtonIteration
from np_Newton_Raphson_Iteration import NewtonIteration
# from function_tools import  Mc_bracket
class ChabocheModel1D(object):

    def __init__(self, E, sgm0, c1, c2, c3, c4, r1, r2, r3, r4, bios1, Q1, bios2, Q2, bios3):   ## 十个参数
        """
        Args:
            param1 (array):Array containing the material parameters

        """
        # 控制两个短程背应力演化速度和短程弹性抗力演化速率相同，可以在此修改或调解该参数限制。

        # *****************
        self.E = E                    # 弹性模量 数值
        self.v = 0.3                        # 泊松比
        self.sgm0 = sgm0              # 屈服强度


        # self.c1 = c1                  # 背应力分量
        # self.c2 = c2                  # 背应力分量
        # self.c3 = c3                  # 背应力分量
        # self.c4 = c4
        self.ci = np.array([c1, c2, c3, c4])

        # self.r1 = r1                  # 背应力分量
        # self.r2 = r2                  # 背应力分量
        # self.r3 = r3                  # 背应力分量
        # self.r4 = r4
        self.ri = np.array([r1, r2, r3, r4])

        # self.bios1 = bios1              # 各向同性演化速率控制参数
        # self.Q1 = Q1                    # 各向同性硬化饱和参数
        # self.bios2 = bios2              # 各向同性演化速率控制参数
        # self.Q2 = Q2                    # 各向同性硬化饱和参数
        # self.bios3 = bios3              # 各向同性演化速率控制参数
        self.biosi = np.array([bios1, bios2])
        self.bios3 = bios3
        self.Qi = np.array([Q1, Q2])

        self.G = self.E / (2 * (1 + self.v))     # 剪切模量


    #### 一下变量以列表形式存储，初始化值为 0 ，但应注意的是，迭代计算后的结果第一位 0 无意义，需要手动删除后计算适应度函数！！！
        # self.ai = np.zeros((4,1),)
        # self.da = np.zeros((4,1),)
        # self.dRi = np.zeros((3,1))
        # self.Ri = np.zeros((3,1))
        # self.a1 = np.array([0])                  # 记录每个加载步的背应力分量a1
        # self.a2 = np.array([0])                  # 记录每个加载步的背应力分量a2
        # self.a3 = np.array([0])                  # 记录每个加载步的背应力分量a3
        # self.a4 = np.array([0])
        # self.a = np.array([0])                   # 记录每个加载步的背应力a
        self.a1 = [0]                  # 记录每个加载步的背应力分量a1
        self.a2 = [0]                  # 记录每个加载步的背应力分量a2
        self.a3 = [0]                  # 记录每个加载步的背应力分量a3
        self.a4 = [0]
        self.a = [0]                   # 记录每个加载步的背应力a

        # self.da1 = np.array([0])
        # self.da2 = np.array([0])
        # self.da3 = np.array([0])
        # self.da4 = np.array([0])
        self.da1 = []
        self.da2 = []
        self.da3 = []
        self.da4 = []
        self.da = []

        # self.dR1 = np.array([0])                   # 记录每个加载步的弹性域大小增量 dR
        # self.R1 = np.array([0])                   # 记录每个加载步的弹性域大小R
        # self.dR2 = np.array([0])                   # 记录每个加载步的弹性域大小增量 dR
        # self.R2 = np.array([0])                   # 记录每个加载步的弹性域大小R
        # self.dR3 = np.array([0])                   # 记录每个加载步的弹性域大小增量 dR
        # self.R3 = np.array([0])                   # 记录每个加载步的弹性域大小R
        # self.dR1 = [0]                   # 记录每个加载步的弹性域大小增量 dR
        self.R1 = [0]
        self.dR1 = []                   # 记录每个加载步的弹性域大小R
        self.dR2 = []                   # 记录每个加载步的弹性域大小增量 dR
        self.R2 = [0]                   # 记录每个加载步的弹性域大小R
        self.dR3 = []                   # 记录每个加载步的弹性域大小增量 dR
        self.R3 = [0]                   # 记录每个加载步的弹性域大小R
        self.dR = []
        self.R = [0]

        self.p = [0]                   # 记录每个加载步的累积塑性应变p
        self.dp = []                  # 记录每个加载步的累积塑性应变增量dp
        self.dpeps = []               # 记录每个加载步的塑性应变增量 dpeps
        self.deeps = []
        self.sgm = [0]                 # 记录每个加载步的应力 sgm
        self.peps = [0]                # 记录每个加载步的塑性应变值peps
        self.eeps = [0]

        # self.p = np.array([0])                   # 记录每个加载步的累积塑性应变p
        # self.dp = np.array([0])                  # 记录每个加载步的累积塑性应变增量dp
        # self.dpeps = np.array([0])               # 记录每个加载步的塑性应变增量 dpeps
        # self.deeps = np.array([0])
        # self.sgm = np.array([0])                 # 记录每个加载步的应力 sgm
        # self.peps = np.array([0])                # 记录每个加载步的塑性应变值peps
        # self.eeps = np.array([0])

    def step_solve(self, teps_n1, teps_n):
        ## 该模块根据当前状态完成一个增量步的计算并更新信息

        '''
        Args:
            param1: ChabocheModelID object ChabocheModel1D 类对象 以类对象为参数不好，容易造成内存过大~~~~~数据传输缓慢
            param2: the total strain value (n+1th step)  teps 第n+1步的总应变
        '''
        ### 将上一步（第n步计算的状态变量结果提取出来，以便进行新的迭代）
        # ai = self.ai[:, -1]
        # a = self.a[-1]

        # a1 = self.a1[-1]
        # a2 = self.a2[-1]
        # a3 = self.a3[-1]
        # a4 = self.a4[-1]
        # ai = np.array([self.a1[-1], self.a2[-1], self.a3[-1] ,self.a4[-1]])
        # a = self.a[-1]
        ai_n = np.array([self.a1[-1], self.a2[-1], self.a3[-1] ,self.a4[-1]])
        a_n = self.a[-1]

        # dai = self.dai[:, -1]
        # da = self.da[-1]
        # da1 = self.da1[-1]
        # da2 = self.da2[-1]
        # da3 = self.da3[-1]
        # da4 = self.da4[-1]
        # dai = np.array([self.da1[-1],self.da2[-1],self.da3[-1]],self.da4[-1])
        # da = self.da[-1]

        # dRi = self.dRi[:, -1]
        # dR = self.dR[-1]
        # Ri = self.Ri[:, -1]
        # R = self.R[-1]
        # dR1 = self.dR1[-1]
        # dR2 = self.dR2[-1]
        # dRi = np.array([self.dR1[-1], self.dR2[-1]])
        # dR3 = self.dR3[-1]
        # dR = self.dR[-1]

        # R1 = self.R1[-1]
        # R2 = self.R2[-1]
        # Ri = np.array([self.R1[-1], self.R2[-1]])
        # R3 = self.R3[-1]
        # R = self.R[-1]
        Ri_n = np.array([self.R1[-1], self.R2[-1]])
        R3_n = self.R3[-1]
        R_n = self.R[-1]

        # dp = self.dp[-1]
        p_n = self.p[-1]
        peps_n = self.peps[-1]
        sgm_n = self.sgm[-1]
        # dpeps = self.dpeps[-1]
        # peps = self.peps[-1]
        # sgm = self.sgm[-1]
        # deeps = self.deeps[-1]
        # eeps = self.eeps[-1]
        eeps_n = self.eeps[-1]
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
        F_yield = abs(stress_trial - a_n) - self.sgm0 - R_n


        if (F_yield <= ERR): # 判断为弹性状态
            sgm_n1 = stress_trial
            deeps_n1 = teps_n1 - teps_n    # 弹性应变增量
            eeps_n1 = eeps_n + deeps_n1

            self.a1.append(ai_n[0])
            self.a2.append(ai_n[1])
            self.a3.append(ai_n[2])
            self.a4.append(ai_n[3])
            self.a.append(a_n)

            self.da1.append(0)
            self.da2.append(0)
            self.da3.append(0)
            self.da4.append(0)
            self.da.append(0)

            self.R1.append(Ri_n[0])
            self.R2.append(Ri_n[1])
            self.R3.append(R3_n)
            self.R.append(R_n)

            self.dR1.append(0)
            self.dR2.append(0)
            self.dR3.append(0)
            self.dR.append(0)

            self.dp.append(0)
            self.p.append(p_n)
            self.dpeps.append(0)
            self.peps.append(peps_n)


            self.deeps.append(deeps_n1)
            self.eeps.append(eeps_n1)

            self.sgm.append(sgm_n1)

            # self.a1.append(ai[0])
            # self.a2.append(ai[1])
            # self.a3.append(ai[2])
            # self.a4.append(ai[3])
            # self.a.append(a)

            # self.da1.append(0)
            # self.da2.append(0)
            # self.da3.append(0)
            # self.da4.append(0)
            # self.da.append(0)

            # self.R1.append(Ri[0])
            # self.R2.append(Ri[1])
            # self.R3.append(R3)
            # self.R.append(R)

            # self.dR1.append(0)
            # self.dR2.append(0)
            # self.dR3.append(0)
            # self.dR.append(0)

            # self.dp.append(0)
            # self.p.append(p)
            # self.dpeps.append(0)


            # self.peps.append(peps)


            # self.deeps.append(deeps)
            # self.eeps.append(eeps)

            # self.sgm.append(sgm)
            # self.a1 = np.append(self.a1, a1)
            # self.a2 = np.append(self.a2, a2)
            # self.a3 = np.append(self.a3, a3)
            # self.a4 = np.append(self.a4, a4)
            # self.a = np.append(self.a, a)

            # self.da1 = np.append(self.da1, 0)
            # self.da2 = np.append(self.da2, 0)
            # self.da3 = np.append(self.da3, 0)
            # self.da4 = np.append(self.da4, 0)
            # self.da = np.append(self.da, 0)

            # self.R1 = np.append(self.R1, R1)
            # self.R2 = np.append(self.R2, R2)
            # self.R3 = np.append(self.R3, R3)
            # self.R = np.append(self.R, R)

            # self.dR1 = np.append(self.dR1, 0)
            # self.dR2 = np.append(self.dR2, 0)
            # self.dR3 = np.append(self.dR3, 0)
            # self.dR = np.append(self.dR, 0)

            # self.dp = np.append(self.dp, 0)
            # self.p = np.append(self.p, p)
            # self.dpeps = np.append(self.dpeps, 0)
            # self.peps = np.append(self.peps, peps)
            # self.deeps = np.append(self.deeps, deeps)
            # self.eeps = np.append(self.eeps, eeps)

            # self.sgm = np.append(self.sgm, sgm)

        else:        # 进入塑性状态
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
            # self.a1.append(ai[0])
            # self.a2.append(ai[1])
            # self.a3.append(ai[2])
            # self.a4.append(ai[3])
            # self.a.append(a)

            # self.a1 = np.append(self.a1, a1)
            # self.a2 = np.append(self.a2, a2)
            # self.a3 = np.append(self.a3, a3)
            # self.a4 = np.append(self.a4, a4)
            # self.a = np.append(self.a, a)

            self.da1.append(dai_n1[0])
            self.da2.append(dai_n1[1])
            self.da3.append(dai_n1[2])
            self.da4.append(dai_n1[3])
            self.da.append(da_n1)
            # self.da1.append(dai[0])
            # self.da2.append(dai[1])
            # self.da3.append(dai[2])
            # self.da4.append(dai[3])
            # self.da.append(da)
            # self.da1 = np.append(self.da1, da1)
            # self.da2 = np.append(self.da2, da2)
            # self.da3 = np.append(self.da3, da3)
            # self.da4 = np.append(self.da4, da4)
            # self.da = np.append(self.da, da)
            self.R1.append(Ri_n1[0])
            self.R2.append(Ri_n1[1])
            self.R3.append(R3_n1)
            self.R.append(R_n1)
            # self.R1.append(Ri[0])
            # self.R2.append(Ri[1])
            # self.R3.append(R3)
            # self.R.append(R)
            # self.R1 = s.append(self.R1, R1)
            # self.R2 = np.append(self.R2, R2)
            # self.R3 = np.append(self.R3, R3)
            # self.R = np.append(self.R, R)

            self.dR1.append(dRi_n1[0])
            self.dR2.append(dRi_n1[1])
            self.dR3.append(dR3_n1)
            self.dR.append(dR_n1)
            # self.dR1.append(dRi[0])
            # self.dR2.append(dRi[1])
            # self.dR3.append(dR3)
            # self.dR.append(dR)
            # self.dR1 = np.append(self.dR1, dR1)
            # self.dR2 = np.append(self.dR2, dR2)
            # self.dR3 = np.append(self.dR3, dR3)
            # self.dR = np.append(self.dR, dR)

            self.dp.append(dpn1)
            # self.p.append(dpn1)
            self.p.append(p_n1)
            self.dpeps.append(dpeps_n1)
            self.peps.append(peps_n1)
            self.deeps.append(deeps_n1)
            self.eeps.append(eeps_n1)
            self.sgm.append(sgm_n1)
            # self.p.append(p)
            # self.dpeps.append(dpeps)
            # self.peps.append(peps)
            # self.deeps.append(deeps)
            # self.eeps.append(eeps)
            # self.dp = np.append(self.dp, dpn1)
            # self.p = np.append(self.p, p)
            # self.dpeps = np.append(self.dpeps, dpeps)
            # self.peps = np.append(self.peps, peps)
            # self.deeps = np.append(self.deeps, deeps)
            # self.eeps = np.append(self.eeps, eeps)
            # self.sgm.append(sgm)
            # self.sgm = np.append(self.sgm, sgm)



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

        # del self.da1[0]
        # del self.da2[0]
        # del self.da3[0]
        # del self.da4[0]
        # del self.da[0]

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

        # del self.dpeps[0]
        # del self.dp[0]
        # del self.deeps[0]

        # self.sgm = np.delete(self.sgm, 0, 0)

        # self.a1 = np.delete(self.a1, 0, 0)
        # self.a2 = np.delete(self.a2, 0, 0)
        # self.a3 = np.delete(self.a3, 0, 0)
        # self.a4 = np.delete(self.a4, 0, 0)
        # self.a = np.delete(self.a, 0, 0)

        # self.da1 = np.delete(self.da1, 0, 0)
        # self.da2 = np.delete(self.da2, 0, 0)
        # self.da3 = np.delete(self.da3, 0, 0)
        # self.da4 = np.delete(self.da4, 0, 0)
        # self.da = np.delete(self.da, 0, 0)

        # self.R1 = np.delete(self.R1, 0, 0)
        # self.R2 = np.delete(self.R2, 0, 0)
        # self.R3 = np.delete(self.R3, 0, 0)
        # self.R = np.delete(self.R, 0, 0)

        # self.dR1 = np.delete(self.dR1, 0, 0)
        # self.dR2 = np.delete(self.dR2, 0, 0)
        # self.dR3 = np.delete(self.dR3, 0, 0)
        # self.dR = np.delete(self.dR, 0, 0)

        # self.peps = np.delete(self.peps, 0, 0)
        # self.dpeps = np.delete(self.dpeps, 0, 0)
        # self.p = np.delete(self.p, 0, 0)
        # self.dp = np.delete(self.dp, 0, 0)
        # self.deeps = np.delete(self.deeps, 0, 0)
        # self.eeps = np.delete(self.eeps, 0, 0)

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
