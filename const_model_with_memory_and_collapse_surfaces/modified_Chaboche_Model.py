# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 20:20:27 2021

@author: SZheng
"""

'''
Author: your name
Date: 2021-04-04 20:19:33
LastEditTime: 2021-04-10 18:11:03
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \Constitutive_Model_parameter_optimization\ChaobocheModel1D.py
'''
import numpy as np
from modif_Newton_Raphson_Iteration import NewtonIteration
# from function_tools import  Mc_bracket
class ChabocheModel1D(object):

    def __init__(self, E, sgm0, c1, c2, c3, r1, r2, bios, Q):   ## 十个参数
        """
        Args:
            param1 (array):Array containing the material parameters

        """
        self.E = E                    # 弹性模量 数值
        self.v = 0.3                        # 泊松比
        self.sgm0 = sgm0              # 屈服强度
        self.c1 = c1                  # 背应力分量
        self.c2 = c2                  # 背应力分量
        self.c3 = c3                  # 背应力分量
        self.r1 = r1                  # 背应力分量
        self.r2 = r2                  # 背应力分量
        self.r3 = 0                  # 背应力分量
        self.bios = bios              # 各向同性演化速率控制参数
        self.Q =  Q                    # 各向同性硬化饱和参数
        self.G = self.E/(2 *(1 + self.v))     # 剪切模量


    #### 一下变量以列表形式存储，初始化值为 0 ，但应注意的是，迭代计算后的结果第一位 0 无意义，需要手动删除后计算适应度函数！！！
        self.a1 = [0]                  # 记录每个加载步的背应力分量a1
        self.a2 = [0]                  # 记录每个加载步的背应力分量a2
        self.a3 = [0]                  # 记录每个加载步的背应力分量a3
        self.da1 = []
        self.da2 = []
        self.da3 = []
        self.da = []
        self.a = [0]                   # 记录每个加载步的背应力a
        self.dR = []                   # 记录每个加载步的弹性域大小增量 dR
        self.R = [0]                   # 记录每个加载步的弹性域大小R
        self.p = [0]                   # 记录每个加载步的累积塑性应变p
        self.dp = []                  # 记录每个加载步的累积塑性应变增量dp
        self.dpeps = []               # 记录每个加载步的塑性应变增量 dpeps
        self.deeps = []
        self.sgm = [0]                 # 记录每个加载步的应力 sgm
        self.peps = [0]                # 记录每个加载步的塑性应变值peps
        self.eeps = [0]

    def step_solve(self, teps_n1,teps_n):
        ## 该模块根据当前状态完成一个增量步的计算并更新信息

        '''
        Args:
            param1: ChabocheModelID object ChabocheModel1D 类对象 以类对象为参数不好，容易造成内存过大~~~~~数据传输缓慢
            param2: the total strain value (n+1th step)  teps 第n+1步的总应变
        '''
        ### 将上一步（第n步计算的状态变量结果提取出来，以便进行新的迭代）
        a1 = self.a1[-1]
        a2 = self.a2[-1]
        a3 = self.a3[-1]
        a = self.a[-1]
        # dR = self.dR[-1]
        R = self.R[-1]
        # dp = self.dp[-1]
        p = self.p[-1]
        # dpeps = self.dpeps[-1]
        peps = self.peps[-1]
        sgm = self.sgm[-1]
        # deeps = self.deeps[-1]
        eeps = self.eeps[-1]
        # 根据第n+1步加载应变计算试应力！
        stress_trial = sgm + self.E * (teps_n1 - teps_n)
        # E_eff = self.E * (1 - self.E / (self.E + self.c1 + self.c2 +\
        #                 self.c3 - self.r1 * a1 - self.r2 * a2 - self.r3 * a3 +\
        #                     self.bios * (self.Q - R)))
        # stress_trial = sgm + E_eff * (teps_n1-teps_n)

        # stress_trial = self.E * (teps_n1 - teps_n + eeps)                   # 计算第 n+1步 试应力 n+1步总应变-n步塑性应变
        ERR = 0
        F_yield = abs(stress_trial - a1 - a2 - a3) - self.sgm0 - R

        if (F_yield <= ERR): # 判断为弹性状态
            sgm = stress_trial
            deeps = teps_n1 - teps_n    # 弹性应变增量
            eeps = eeps + deeps
                                                    # 弹性域增量
                                                    # a1增量
                                                    # a2增量
                                                    # a3增量
                                                    # 塑性应变增量
                                                    # 累积塑性应变增量
                                # 记录每一步弹性计算的中间变量结果
            self.a1.append(a1)
            self.a2.append(a2)
            self.a3.append(a3)
            self.a.append(a)

            self.da1.append(0)
            self.da2.append(0)
            self.da3.append(0)
            self.da.append(0)

            self.sgm.append(sgm)
            self.dR.append(0)
            self.R.append(R)
            self.dp.append(0)
            self.p.append(p)
            self.dpeps.append(0)
            self.peps.append(peps)
            self.deeps.append(deeps)
            self.eeps.append(eeps)


        else:        # 进入塑性状态
                    # 通过Newton-Raphson迭代求解第 n+1步 产生的等效塑性应变dp
            dpn1 = NewtonIteration(a1,a2,a3,self.c1,self.c2,self.c3,self.r1,self.r2,self.r3,\
                self.G,self.sgm0,self.Q,R,self.bios,stress_trial)      # 调用dp求解函数迭代求解该加载步的dp
                    # 由dp值，分别更新 dpeps，peps，da1，da2，da3，a1，a2，a3，a，dR，R，sgm
            theta1 = 1 / (1 + self.r1 * dpn1)
            theta2 = 1 / (1 + self.r2 * dpn1)
            theta3 = 1 / (1 + self.r3 * dpn1)
            stress_trial_n1 = stress_trial - (theta1*a1 + theta2*a2 +theta3*a3)
            dpeps = np.sign(stress_trial_n1) * dpn1  ## 计算这一步的塑性应变增量
            peps = peps + dpeps     ## 更新本步的总塑性应变 n+1步的值
            deeps = teps_n1 - teps_n - dpeps
            eeps = eeps + deeps
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
            da1 = (self.c1*dpeps - self.r1*a1*dpn1)/(1 + self.r1 * dpn1)             # 计算背应力分量增量
            da2 = (self.c2*dpeps - self.r2*a2*dpn1)/(1 + self.r2 * dpn1)
            da3 = (self.c3*dpeps - self.r3*a3*dpn1)/(1 + self.r3 * dpn1)
            da = da1 + da2 + da3

            a1 = (a1 + self.c1*dpeps)/(1 + self.r1 * dpn1)
            a2 = (a2 + self.c2*dpeps)/(1 + self.r2 * dpn1)
            a3 = (a3 + self.c3*dpeps)/(1 + self.r3 * dpn1)
            # a1 = (a1 + da1)
            # a2 = (a2 + da2)
            # a3 = (a3 + da3)
            a = a1 + a2 + a3

            # dR = (self.bios * Mc_bracket(self.Q-R) * dp)/(1+self.bios * dp)
            # dR = (self.bios * (self.Q-R) * dp)/(1+self.bios * dp)
            dR = (self.bios * (self.Q - R) * dpn1)/(1 + self.bios * dpn1)
            R = (R + self.bios * self.Q * dpn1)/(1 + self.bios * dpn1)
            # R = R + dR
            p = p + dpn1


            # sgm = stress_trial - 2 * self.G * dpeps # 根据第n+1步的弹性应变更新应力，而不是直接按照总的计算，好像可以减少步计算的误差???
            sgm = sgm + self.E * deeps
            # sgm = sgm + E_eff * (teps_n1 - teps_n)

            self.a1.append(a1)
            self.a2.append(a2)
            self.a3.append(a3)
            self.a.append(a)

            self.da1.append(da1)
            self.da2.append(da2)
            self.da3.append(da3)
            self.da.append(da)

            self.dp.append(dpn1)
            self.p.append(p)
            self.dR.append(dR)
            self.R.append(R)
            self.sgm.append(sgm)
            self.dpeps.append(dpeps)
            self.peps.append(peps)
            self.deeps.append(deeps)
            self.eeps.append(eeps)

    def total_solve(self, l_strain):       ## l_strain:list[],即接受一个加载的试验应变列表，通过迭代循环求解每一步的计算结果
        # 列表类型是可变的，接受列表参数可对列表直接进行修改，不需要返回值。
        self.step_solve(l_strain[0],0)


        for i in range(1,len(l_strain)):
            self.step_solve(l_strain[i],l_strain[i-1])    ## 直接修改了该类ChabocheModel对象的self属性值！
        ## 删除列表初始化造成的无用初值 0
        del self.sgm[0]
        del self.a1[0]
        del self.a2[0]
        del self.a3[0]
        del self.a[0]
        del self.peps[0]
        del self.p[0]
        del self.R[0]
        del self.eeps[0]
        return self.sgm


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
