# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 15:49:12 2021

@author: SZheng
"""

import numpy as np
def sum_exp2(l):
    sum_exp = 0
    for i in range(len(l)):
        sum_exp = sum_exp + l[i]**2
    return sum_exp

def sum_min_exp_Num(a,b,Num_1,Num_2): ## 包括Num_1但不包括Num_2
    # 将列表a，b中的第[Num_1,Num_2)位元素值，求差方和，b的平方和，并返回值
    sum_min_square = 0
    b_sum_square = 0
    if (Num_1 > len(a) or Num_1 > len(b) or Num_2 > len(a) or Num_2 > len(b) or Num_1 > Num_2):
        raise ValueError
    for i in range(Num_1,Num_2):
        sum_min_square = sum_min_square + (a[Num_1]-b[Num_1])**2
        b_sum_square = b_sum_square + b[Num_1]**2
    return sum_min_square,b_sum_square
def sum_l(l):
    total = 0
    for i in range(len(l)):
        total += l[i]
    return total

def goal_function(a,b,strain_data):
    # 接受理论模型计算的应力值、试验应力值和试验应变值列表
    Num_a = len(a)
    Num_b = len(b)
    if Num_a != Num_b:
        print('两个应力列表容量不相同，报错！')
        raise ValueError
    # weight_coef = [0,1/1.1,1/1.1,1/1.1,1/1.1,1/1.1]
    # weight_coef[]权重矩阵系数：
    # 其含义分别为：
    # 1：前1/4圈   0.85
    # 2：前几圈     1.1
    # 3：中间几圈    1.0
    # 4：后几圈     1.15
    # 5：每个加载圈的峰值应力附近范围：1.1
    #weights_coef即为衡量不同阶段数据点的适应度/目标函数值权重系数矩阵 weight matrix
    '''
    strain_limit_c_1 = -0.0190
    strain_limit_c_2 = -0.0150
    strain_limit_c_3 = -0.010
    strain_limit_c_4 = -0.005
    strain_limit_t_1 = 0.005
    strain_limit_t_2 = 0.010
    strain_limit_t_3 = 0.015
    strain_limit_t_4 = 0.0185
    '''
    strain_limit_l = [-0.0185,-0.0150,-0.0100,-0.0050,0,0.0050,0.0100,0.0150,0.0185]
    quart_cycle_num = 101
    first_cycle_num = 3750
    mid_cycle_num = 8081
    last_cycle_num = Num_a
    
    # 小应变范围内的差方和
    sum_min_square_quart_value_1 = 0
    sum_min_square_first_value_1 = 0
    sum_min_square_mid_value_1 =0
    sum_min_square_last_value_1 =0

    # 大应变范围(接近峰值应力)内的差方和
    sum_min_square_quart_value_2 = 0
    sum_min_square_first_value_2 = 0
    sum_min_square_mid_value_2 =0
    sum_min_square_last_value_2 =0
    
    sum_min_square_quart_value_3 = 0
    sum_min_square_first_value_3 = 0
    sum_min_square_mid_value_3 = 0
    sum_min_square_last_value_3 = 0
    
    sum_min_square_quart_value_4 = 0
    sum_min_square_first_value_4 = 0
    sum_min_square_mid_value_4 = 0
    sum_min_square_last_value_4 = 0
    
    sum_min_square_quart_value_5 = 0
    sum_min_square_first_value_5 = 0
    sum_min_square_mid_value_5 = 0
    sum_min_square_last_value_5 = 0
    
    sum_min_square_quart_value_6 = 0
    sum_min_square_first_value_6 = 0
    sum_min_square_mid_value_6 = 0
    sum_min_square_last_value_6 = 0
    
    sum_min_square_quart_value_7 = 0
    sum_min_square_first_value_7 = 0
    sum_min_square_mid_value_7 = 0
    sum_min_square_last_value_7 = 0
    
    sum_min_square_quart_value_8 = 0
    sum_min_square_first_value_8 = 0
    sum_min_square_mid_value_8 = 0
    sum_min_square_last_value_8 = 0
    
    sum_min_square_quart_value_9 = 0
    sum_min_square_first_value_9 = 0
    sum_min_square_mid_value_9 = 0
    sum_min_square_last_value_9 = 0
    
    sum_min_square_quart_value_10 = 0
    sum_min_square_first_value_10 = 0
    sum_min_square_mid_value_10 = 0
    sum_min_square_last_value_10 = 0
    # 分母 试验应力列表平方和
    sum_b_square_quart_value_1 = 0
    sum_b_square_first_value_1 = 0
    sum_b_square_mid_value_1 =0
    sum_b_square_last_value_1 =0

    sum_b_square_quart_value_2 = 0
    sum_b_square_first_value_2 = 0
    sum_b_square_mid_value_2 =0
    sum_b_square_last_value_2 =0
    
    sum_b_square_quart_value_3 = 0
    sum_b_square_first_value_3 = 0
    sum_b_square_mid_value_3 = 0
    sum_b_square_last_value_3 = 0
    
    sum_b_square_quart_value_4 = 0
    sum_b_square_first_value_4 = 0
    sum_b_square_mid_value_4 = 0
    sum_b_square_last_value_4 = 0
    
    sum_b_square_quart_value_5 = 0
    sum_b_square_first_value_5 = 0
    sum_b_square_mid_value_5 = 0
    sum_b_square_last_value_5 = 0
    
    sum_b_square_quart_value_6 = 0
    sum_b_square_first_value_6 = 0
    sum_b_square_mid_value_6 = 0
    sum_b_square_last_value_6 = 0
    
    sum_b_square_quart_value_7 = 0
    sum_b_square_first_value_7 = 0
    sum_b_square_mid_value_7 = 0
    sum_b_square_last_value_7 = 0
    
    sum_b_square_quart_value_8 = 0
    sum_b_square_first_value_8 = 0
    sum_b_square_mid_value_8 = 0
    sum_b_square_last_value_8 = 0
    
    sum_b_square_quart_value_9 = 0
    sum_b_square_first_value_9 = 0
    sum_b_square_mid_value_9 = 0
    sum_b_square_last_value_9 = 0
    
    sum_b_square_quart_value_10 = 0
    sum_b_square_first_value_10 = 0
    sum_b_square_mid_value_10 = 0
    sum_b_square_last_value_10 = 0
    
    l_a_b_square_quart = []
    l_a_b_square_first = []
    l_a_b_square_mid = []
    l_a_b_square_last = []

    l_b_square_quart = []
    l_b_square_first = []
    l_b_square_mid = []
    l_b_square_last = []
    for i in range(quart_cycle_num):
        if (strain_data[i] > strain_limit_l[4]) & (strain_data[i] < strain_limit_l[5]):   # 前1/4圈
            sum_min_square_quart_value_6 += (a[i]-b[i])**2
            sum_b_square_quart_value_6 += b[i]**2
        if (strain_data[i] > strain_limit_l[5]) & (strain_data[i] < strain_limit_l[6]):
            sum_min_square_quart_value_7 += (a[i]-b[i])**2
            sum_b_square_quart_value_7 += b[i]**2
        if (strain_data[i] > strain_limit_l[6]) & (strain_data[i] < strain_limit_l[7]):
            sum_min_square_quart_value_8 += (a[i]-b[i])**2
            sum_b_square_quart_value_8 += b[i]**2
        if (strain_data[i] > strain_limit_l[7]) & (strain_data[i] < strain_limit_l[8]):
            sum_min_square_quart_value_9 += (a[i]-b[i])**2
            sum_b_square_quart_value_9 += b[i]**2
        if (strain_data[i] > strain_limit_l[8]):
            sum_min_square_quart_value_10 += (a[i]-b[i])**2
            sum_b_square_quart_value_10 += b[i]**2
    l_a_b_square_quart.append(sum_min_square_quart_value_1)
    l_a_b_square_quart.append(sum_min_square_quart_value_2)
    l_a_b_square_quart.append(sum_min_square_quart_value_3)
    l_a_b_square_quart.append(sum_min_square_quart_value_4)
    l_a_b_square_quart.append(sum_min_square_quart_value_5)
    l_a_b_square_quart.append(sum_min_square_quart_value_6)
    l_a_b_square_quart.append(sum_min_square_quart_value_7)
    l_a_b_square_quart.append(sum_min_square_quart_value_8)
    l_a_b_square_quart.append(sum_min_square_quart_value_9)
    l_a_b_square_quart.append(sum_min_square_quart_value_10)
    l_b_square_quart.append(sum_b_square_quart_value_1)
    l_b_square_quart.append(sum_b_square_quart_value_2)
    l_b_square_quart.append(sum_b_square_quart_value_3)
    l_b_square_quart.append(sum_b_square_quart_value_4)
    l_b_square_quart.append(sum_b_square_quart_value_5)
    l_b_square_quart.append(sum_b_square_quart_value_6)
    l_b_square_quart.append(sum_b_square_quart_value_7)
    l_b_square_quart.append(sum_b_square_quart_value_8)
    l_b_square_quart.append(sum_b_square_quart_value_9)
    l_b_square_quart.append(sum_b_square_quart_value_10)

    for j in range(quart_cycle_num,first_cycle_num):
        if (strain_data[j] < strain_limit_l[0]):
            sum_min_square_first_value_1 += (a[j]-b[j])**2
            sum_b_square_first_value_1 += b[j]**2
        if (strain_data[j] > strain_limit_l[0]) & (strain_data[j] < strain_limit_l[1]):
            sum_min_square_first_value_2 += (a[j]-b[j])**2
            sum_b_square_first_value_2 += b[j]**2
        if (strain_data[j] > strain_limit_l[1]) & (strain_data[j] < strain_limit_l[2]):
            sum_min_square_first_value_3 += (a[j]-b[j])**2
            sum_b_square_first_value_3 += b[j]**2
        if (strain_data[j] > strain_limit_l[2]) & (strain_data[j] < strain_limit_l[3]):
            sum_min_square_first_value_4 += (a[j]-b[j])**2
            sum_b_square_first_value_4 += b[j]**2
        if (strain_data[j] > strain_limit_l[3]) & (strain_data[j] < strain_limit_l[4]):
            sum_min_square_first_value_5 += (a[j]-b[j])**2
            sum_b_square_first_value_5 += b[j]**2
        if (strain_data[j] > strain_limit_l[4]) & (strain_data[j] < strain_limit_l[5]):
            sum_min_square_first_value_6 += (a[j]-b[j])**2
            sum_b_square_first_value_6 += b[j]**2
        if (strain_data[j] > strain_limit_l[5]) & (strain_data[j] < strain_limit_l[6]):
            sum_min_square_first_value_7 += (a[j]-b[j])**2
            sum_b_square_first_value_7 += b[j]**2
        if (strain_data[j] > strain_limit_l[6]) & (strain_data[j] < strain_limit_l[7]):
            sum_min_square_first_value_8 += (a[j]-b[j])**2
            sum_b_square_first_value_8 += b[j]**2
        if (strain_data[j] > strain_limit_l[7]) & (strain_data[j] < strain_limit_l[8]):
            sum_min_square_first_value_9 += (a[j]-b[j])**2
            sum_b_square_first_value_9 += b[j]**2
        if (strain_data[j] > strain_limit_l[8]):
            sum_min_square_first_value_10 += (a[j]-b[j])**2
            sum_b_square_first_value_10 += b[j]**2
    l_a_b_square_first.append(sum_min_square_first_value_1)
    l_a_b_square_first.append(sum_min_square_first_value_2)
    l_a_b_square_first.append(sum_min_square_first_value_3)
    l_a_b_square_first.append(sum_min_square_first_value_4)
    l_a_b_square_first.append(sum_min_square_first_value_5)
    l_a_b_square_first.append(sum_min_square_first_value_6)
    l_a_b_square_first.append(sum_min_square_first_value_7)
    l_a_b_square_first.append(sum_min_square_first_value_8)
    l_a_b_square_first.append(sum_min_square_first_value_9)
    l_a_b_square_first.append(sum_min_square_first_value_10)
    l_b_square_first.append(sum_b_square_first_value_1)
    l_b_square_first.append(sum_b_square_first_value_2)
    l_b_square_first.append(sum_b_square_first_value_3)
    l_b_square_first.append(sum_b_square_first_value_4)
    l_b_square_first.append(sum_b_square_first_value_5)
    l_b_square_first.append(sum_b_square_first_value_6)
    l_b_square_first.append(sum_b_square_first_value_7)
    l_b_square_first.append(sum_b_square_first_value_8)
    l_b_square_first.append(sum_b_square_first_value_9)
    l_b_square_first.append(sum_b_square_first_value_10)

    for k in range(first_cycle_num,mid_cycle_num):  # 中间圈
        if (strain_data[k] < strain_limit_l[0]):
            sum_min_square_mid_value_1 += (a[k]-b[k])**2
            sum_b_square_mid_value_1 += b[k]**2
        if (strain_data[k] > strain_limit_l[0]) & (strain_data[k] < strain_limit_l[1]):
            sum_min_square_mid_value_2 += (a[k]-b[k])**2
            sum_b_square_mid_value_2 += b[k]**2
        if (strain_data[k] > strain_limit_l[1]) & (strain_data[k] < strain_limit_l[2]):
            sum_min_square_mid_value_3 += (a[k]-b[k])**2
            sum_b_square_mid_value_3 += b[k]**2
        if (strain_data[k] > strain_limit_l[2]) & (strain_data[k] < strain_limit_l[3]):
            sum_min_square_mid_value_4 += (a[k]-b[k])**2
            sum_b_square_mid_value_4 += b[k]**2
        if (strain_data[k] > strain_limit_l[3]) & (strain_data[k] < strain_limit_l[4]):
            sum_min_square_mid_value_5 += (a[k]-b[k])**2
            sum_b_square_mid_value_5 += b[k]**2
        if (strain_data[k] > strain_limit_l[4]) & (strain_data[k] < strain_limit_l[5]):
            sum_min_square_mid_value_6 += (a[k]-b[k])**2
            sum_b_square_mid_value_6 += b[k]**2
        if (strain_data[k] > strain_limit_l[5]) & (strain_data[k] < strain_limit_l[6]):
            sum_min_square_mid_value_7 += (a[k]-b[k])**2
            sum_b_square_mid_value_7 += b[k]**2
        if (strain_data[k] > strain_limit_l[6]) & (strain_data[k] < strain_limit_l[7]):
            sum_min_square_mid_value_8 += (a[k]-b[k])**2
            sum_b_square_mid_value_8 += b[k]**2
        if (strain_data[k] > strain_limit_l[7]) & (strain_data[k] < strain_limit_l[8]):
            sum_min_square_mid_value_9 += (a[k]-b[k])**2
            sum_b_square_mid_value_9 += b[k]**2
        if (strain_data[k] > strain_limit_l[8]):
            sum_min_square_mid_value_10 += (a[k]-b[k])**2
            sum_b_square_mid_value_10 += b[k]**2
    l_a_b_square_mid.append(sum_min_square_mid_value_1)
    l_a_b_square_mid.append(sum_min_square_mid_value_2)
    l_a_b_square_mid.append(sum_min_square_mid_value_3)
    l_a_b_square_mid.append(sum_min_square_mid_value_4)
    l_a_b_square_mid.append(sum_min_square_mid_value_5)
    l_a_b_square_mid.append(sum_min_square_mid_value_6)
    l_a_b_square_mid.append(sum_min_square_mid_value_7)
    l_a_b_square_mid.append(sum_min_square_mid_value_8)
    l_a_b_square_mid.append(sum_min_square_mid_value_9)
    l_a_b_square_mid.append(sum_min_square_mid_value_10)
    l_b_square_mid.append(sum_b_square_mid_value_1)
    l_b_square_mid.append(sum_b_square_mid_value_2)
    l_b_square_mid.append(sum_b_square_mid_value_3)
    l_b_square_mid.append(sum_b_square_mid_value_4)
    l_b_square_mid.append(sum_b_square_mid_value_5)
    l_b_square_mid.append(sum_b_square_mid_value_6)
    l_b_square_mid.append(sum_b_square_mid_value_7)
    l_b_square_mid.append(sum_b_square_mid_value_8)
    l_b_square_mid.append(sum_b_square_mid_value_9)
    l_b_square_mid.append(sum_b_square_mid_value_10)


    for l in range(mid_cycle_num,last_cycle_num):   # 最后十圈
        if (strain_data[l] < strain_limit_l[0]):
            sum_min_square_last_value_1 += (a[l]-b[l])**2
            sum_b_square_last_value_1 += b[l]**2
        if (strain_data[l] > strain_limit_l[0]) & (strain_data[l] < strain_limit_l[1]):
            sum_min_square_last_value_2 += (a[l]-b[l])**2
            sum_b_square_last_value_2 += b[l]**2
        if (strain_data[l] > strain_limit_l[1]) & (strain_data[l] < strain_limit_l[2]):
            sum_min_square_last_value_3 += (a[l]-b[l])**2
            sum_b_square_last_value_3 += b[l]**2
        if (strain_data[l] > strain_limit_l[2]) & (strain_data[l] < strain_limit_l[3]):
            sum_min_square_last_value_4 += (a[l]-b[l])**2
            sum_b_square_last_value_4 += b[l]**2
        if (strain_data[l] > strain_limit_l[3]) & (strain_data[l] < strain_limit_l[4]):
            sum_min_square_last_value_5 += (a[l]-b[l])**2
            sum_b_square_last_value_5 += b[l]**2
        if (strain_data[l] > strain_limit_l[4]) & (strain_data[l] < strain_limit_l[5]):
            sum_min_square_last_value_6 += (a[l]-b[l])**2
            sum_b_square_last_value_6 += b[l]**2
        if (strain_data[l] > strain_limit_l[5]) & (strain_data[l] < strain_limit_l[6]):
            sum_min_square_last_value_7 += (a[l]-b[l])**2
            sum_b_square_last_value_7 += b[l]**2
        if (strain_data[l] > strain_limit_l[6]) & (strain_data[l] < strain_limit_l[7]):
            sum_min_square_last_value_8 += (a[l]-b[l])**2
            sum_b_square_last_value_8 += b[l]**2
        if (strain_data[l] > strain_limit_l[7]) & (strain_data[l] < strain_limit_l[8]):
            sum_min_square_last_value_9 += (a[l]-b[l])**2
            sum_b_square_last_value_9 += b[l]**2
        if (strain_data[l] > strain_limit_l[8]):
            sum_min_square_last_value_10 += (a[l]-b[l])**2
            sum_b_square_last_value_10 += b[l]**2
    l_a_b_square_last.append(sum_min_square_last_value_1)
    l_a_b_square_last.append(sum_min_square_last_value_2)
    l_a_b_square_last.append(sum_min_square_last_value_3)
    l_a_b_square_last.append(sum_min_square_last_value_4)
    l_a_b_square_last.append(sum_min_square_last_value_5)
    l_a_b_square_last.append(sum_min_square_last_value_6)
    l_a_b_square_last.append(sum_min_square_last_value_7)
    l_a_b_square_last.append(sum_min_square_last_value_8)
    l_a_b_square_last.append(sum_min_square_last_value_9)
    l_a_b_square_last.append(sum_min_square_last_value_10)
    l_b_square_last.append(sum_b_square_last_value_1)
    l_b_square_last.append(sum_b_square_last_value_2)
    l_b_square_last.append(sum_b_square_last_value_3)
    l_b_square_last.append(sum_b_square_last_value_4)
    l_b_square_last.append(sum_b_square_last_value_5)
    l_b_square_last.append(sum_b_square_last_value_6)
    l_b_square_last.append(sum_b_square_last_value_7)
    l_b_square_last.append(sum_b_square_last_value_8)
    l_b_square_last.append(sum_b_square_last_value_9)
    l_b_square_last.append(sum_b_square_last_value_10)

# **************************** 计算考虑权重的目标函数值、以及不考虑权重的目标函数值（计算代价）  *********************************
    # 计算总的适应度(先求和，再计算)
    # 按照分段分别计算各段的适应度函数
    origin_goal_first_i = []    # 存储每段应变应力数据理论值与实验值的均方差
    origin_goal_mid_i = []
    origin_goal_last_i = []
    for i in range(len(strain_limit_l)+1): #按照应变分段数
        per1 = np.sqrt(l_a_b_square_first[i]/l_b_square_first[i])
        origin_goal_first_i.append(per1)
    for j in range(len(strain_limit_l)+1): #按照应变分段数
        per2 = np.sqrt(l_a_b_square_mid[j]/l_b_square_mid[j])
        origin_goal_mid_i.append(per2)
    for k in range(len(strain_limit_l)+1): #按照应变分段数
        per3 = np.sqrt(l_a_b_square_last[k]/l_b_square_last[k])
        origin_goal_last_i.append(per3)
    
    quart_a_b = sum_l(l_a_b_square_quart)
    quart_b = sum_l(l_b_square_quart)
    first_a_b = sum_l(l_a_b_square_first)
    first_b = sum_l(l_b_square_first)
    mid_a_b = sum_l(l_a_b_square_mid)
    mid_b = sum_l(l_b_square_mid)
    last_a_b = sum_l(l_a_b_square_last)
    last_b = sum_l(l_b_square_last)
    
    # 计算总的目标函数值
    origin_goal_value = np.sqrt((quart_a_b+first_a_b+mid_a_b+last_a_b)/(quart_b+first_b+mid_b+last_b))
    
    
    '''
    origin_goal_value = np.sqrt((sum_min_square_quart_value_1+sum_min_square_quart_value_2+\
                                 sum_min_square_first_value_1+sum_min_square_first_value_2+\
                                     sum_min_square_mid_value_1+sum_min_square_mid_value_2+\
                                         sum_min_square_last_value_1+sum_min_square_last_value_2)/\
                                (sum_b_square_quart_value_1+sum_b_square_quart_value_2+\
                                 sum_b_square_first_value_1+sum_b_square_first_value_2+\
                                     sum_b_square_mid_value_1+sum_b_square_mid_value_2+\
                                         sum_b_square_last_value_1+sum_b_square_last_value_2))    #不计算平权目标值了，验算时再计算，节省计算资源
    '''
    '''
    # 计算考虑各个权重系数矩阵的目标函数值，采用分段计算的方法，先按照分段计算出平权值，再乘以各自的权重系数
    weight_goal_value = np.sqrt(sum_min_square_quart_value_1/sum_b_square_quart_value_1)*weight_coef[0]+\
        np.sqrt(sum_min_square_quart_value_2/sum_b_square_quart_value_2)*weight_coef[0]*weight_coef[4]+\
            np.sqrt(sum_min_square_first_value_1/sum_b_square_first_value_1)*weight_coef[1]+\
                np.sqrt(sum_min_square_first_value_2/sum_b_square_first_value_2)*weight_coef[1]*weight_coef[4]+\
                    np.sqrt(sum_min_square_mid_value_1/sum_b_square_mid_value_1)*weight_coef[2]+\
                        np.sqrt(sum_min_square_mid_value_2/sum_b_square_mid_value_2)*weight_coef[2]*weight_coef[4]+\
                            np.sqrt(sum_min_square_last_value_1/sum_b_square_last_value_1)*weight_coef[3]+\
                                np.sqrt(sum_min_square_last_value_2/sum_b_square_last_value_2)*weight_coef[3]*weight_coef[4]+\
                                    origin_goal_value*weight_coef[5]
    '''
    # score_value = (np.sqrt(sum([(x-y)**2.0 for x,y in zip(a,b)])/Num_a))/np.sqrt(sum_exp2(b)/Num_a) ## 考虑会不会由于数据量太大造成大数吃小数的结果？？
        ###fit_value = 1.0/values       # 取理论应力值和试验应力值的均方根倒数作为适应度值，则问题转化为求适应度最大值问题
    # return origin_goal_value,weight_goal_value
    return origin_goal_value,origin_goal_first_i,origin_goal_mid_i,origin_goal_last_i