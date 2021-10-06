# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 14:16:22 2021

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

def goal_function(a,b,strain_data):
    # 接受理论模型计算的应力值、试验应力值和试验应变值列表
    Num_a = len(a)
    Num_b = len(b)
    if Num_a != Num_b:
        print('两个应力列表容量不相同，报错！')
        raise ValueError
    weight_coef = [0,1/1.1,1/1.1,1/1.1,1/1.1,1/1.1]
    # weight_coef[]权重矩阵系数：
    # 其含义分别为：
    # 1：前1/4圈   0.85
    # 2：前几圈     1.1
    # 3：中间几圈    1.0
    # 4：后几圈     1.15
    # 5：每个加载圈的峰值应力附近范围：1.1
    #weights_coef即为衡量不同阶段数据点的适应度/目标函数值权重系数矩阵 weight matrix
    strain_limit_t_1 = 0.005
    strain_limit_c_1 = -0.005
    strain_limit_t_2 = 0.010
    strain_limit_c_2 = -0.010
    strain_limit_t_3 = 0.015
    strain_limit_c_3 = -0.015
    strain_limit_t_4 = 0.0185
    strain_limit_c_4 = -0.0190
    
    
    quart_cycle_num = 100
    first_cycle_num = 3750
    mid_cycle_num = 8081
    last_cycle_num = Num_a
    
    
    """
    list_quart_a = a[0:quart_cycle_num]
    list_first_a = a[quart_cycle_num:first_cycle_num]
    list_mid_a = a[first_cycle_num:mid_cycle_num]
    list_last_a = a[mid_cycle_num:]
    
    list_quart_b = b[0:quart_cycle_num]
    list_first_b = b[quart_cycle_num:first_cycle_num]
    list_mid_b = b[first_cycle_num:mid_cycle_num]
    list_last_b = b[mid_cycle_num:]
    
    quart_squar = sum([(x-y)**2 for x,y in zip(list_quart_a,list_quart_b)])
    b_quart_squar = sum_exp2(list_quart_b)
    
    first_squar = sum([(x-y)**2 for x,y in zip(list_first_a,list_first_b)])
    b_first_squar = sum_exp2(list_first_b)

    mid_squar = sum([(x-y)**2 for x,y in zip(list_mid_a,list_mid_b)])
    b_mid_squar = sum_exp2(list_mid_b)

    last_squar = sum([(x-y)**2 for x,y in zip(list_last_a,list_last_b)])
    b_last_squar = sum_exp2(list_last_b)
    
    # 计算每一段的适应度函数
    """
    '''
    goal_quart_value = 0
    goal_first_value = 0
    goal_mid_value =0
    goal_last_value =0
    '''

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
    
    # 分母 试验应力列表平方和
    sum_b_square_quart_value_1 = 0
    sum_b_square_first_value_1 = 0
    sum_b_square_mid_value_1 =0
    sum_b_square_last_value_1 =0

    sum_b_square_quart_value_2 = 0
    sum_b_square_first_value_2 = 0
    sum_b_square_mid_value_2 =0
    sum_b_square_last_value_2 =0
    
    for i in range(quart_cycle_num):
        if strain_data[i] > strain_limit_t or strain_data[i] < strain_limit_c:   # 前1/4圈
            sum_min_square_quart_value_1 += (a[i]-b[i])**2
            sum_b_square_quart_value_1 += b[i]**2
        else:
            sum_min_square_quart_value_2 += (a[i]-b[i])**2
            # weight_coef[0]权重：第1/4圈小应变范围内的权重系数
            sum_b_square_quart_value_2 += b[i]**2

    for j in range(quart_cycle_num,first_cycle_num):
        if strain_data[j] > strain_limit_t or strain_data[j] < strain_limit_c:   # 1/4圈到前十圈
            sum_min_square_first_value_1 += (a[j]-b[j])**2
            sum_b_square_first_value_1 += b[j]**2
        else:
            sum_min_square_first_value_2 += (a[j]-b[j])**2
            sum_b_square_first_value_2 += b[j]**2

    for k in range(first_cycle_num,mid_cycle_num):  # 中间圈
        if strain_data[k] > strain_limit_t or strain_data[k] < strain_limit_c:   # 前1/4圈
            sum_min_square_mid_value_1 += (a[k]-b[k])**2
            sum_b_square_mid_value_1 += b[k]**2
        else:
            sum_min_square_mid_value_2 += (a[k]-b[k])**2
            sum_b_square_mid_value_2 += b[k]**2


    for l in range(mid_cycle_num,last_cycle_num):   # 最后十圈
        if strain_data[l] > strain_limit_t or strain_data[l] < strain_limit_c:   # 前1/4圈
            sum_min_square_last_value_1 += (a[l]-b[l])**2
            sum_b_square_last_value_1 += b[l]**2
        else:
            sum_min_square_last_value_2 += (a[l]-b[l])**2
            sum_b_square_last_value_2 += b[l]**2

# **************************** 计算考虑权重的目标函数值、以及不考虑权重的目标函数值（计算代价）  *********************************
    
    origin_goal_value = np.sqrt((sum_min_square_quart_value_1+sum_min_square_quart_value_2+\
                                 sum_min_square_first_value_1+sum_min_square_first_value_2+\
                                     sum_min_square_mid_value_1+sum_min_square_mid_value_2+\
                                         sum_min_square_last_value_1+sum_min_square_last_value_2)/\
                                (sum_b_square_quart_value_1+sum_b_square_quart_value_2+\
                                 sum_b_square_first_value_1+sum_b_square_first_value_2+\
                                     sum_b_square_mid_value_1+sum_b_square_mid_value_2+\
                                         sum_b_square_last_value_1+sum_b_square_last_value_2))    #不计算平权目标值了，验算时再计算，节省计算资源
    
    
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
    
    # score_value = (np.sqrt(sum([(x-y)**2.0 for x,y in zip(a,b)])/Num_a))/np.sqrt(sum_exp2(b)/Num_a) ## 考虑会不会由于数据量太大造成大数吃小数的结果？？
        ###fit_value = 1.0/values       # 取理论应力值和试验应力值的均方根倒数作为适应度值，则问题转化为求适应度最大值问题
    return origin_goal_value,weight_goal_value
