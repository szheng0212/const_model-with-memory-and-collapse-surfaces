# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 19:57:40 2021

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

# def goal_function(a,b):
#     # 接受理论模型计算的应力值、试验应力值和试验应变值列表
#     Num_a = len(a)
#     Num_b = len(b)
#     if Num_a != Num_b:
#         print('两个应力列表容量不相同，报错！')
#         raise ValueError
#     total = 0
#     quart = 101
#     for i in range(quart,Num_a):
#         total = total + ((a[i] - b[i]) / b[i])**2
#     goal_value = np.sqrt(total / Num_a)
#     return goal_value

# def goal_function(a,b,strain):
#     # 接受理论模型计算的应力值、试验应力值和试验应变值列表
#     Num_a = len(a)
#     Num_b = len(b)
#     if Num_a != Num_b:
#         print('两个应力列表容量不相同，报错！')
#         raise ValueError
#     w = [1,1.5,1,1.5,1]
#     total_x_y = 0
#     total_y = 0
#     quart = 101
#     strain_l1 = -0.005
#     strain_l2 = -0.015
#     strain_u1 = 0.005
#     strain_u2 = 0.015
#     for i in range(quart,Num_a):
#         if (strain[i] < strain_l2):
#             total_x_y = total_x_y + (a[i]-b[i])**2 * w[0]
#             total_y = total_y + (b[i])**2
#         if ((strain[i] > strain_l2) & (strain[i] < strain_l1)):
#             total_x_y = total_x_y + (a[i]-b[i])**2 * w[1]
#             total_y = total_y + (b[i])**2
#         if ((strain[i] > strain_l1) & (strain[i] < strain_u1)):
#             total_x_y = total_x_y + (a[i]-b[i])**2 * w[2]
#             total_y = total_y + (b[i])**2
#         if ((strain[i] > strain_u1) & (strain[i] < strain_u2)):
#             total_x_y = total_x_y + (a[i]-b[i])**2 * w[3]
#             total_y = total_y + (b[i])**2
#         if (strain[i] > strain_u2):
#             total_x_y = total_x_y + (a[i]-b[i])**2 * w[4]
#             total_y = total_y + (b[i])**2
#         # total = total + ((a[i]-b[i])/b[i])**2
#     goal_value = np.sqrt((total_x_y / total_y) / Num_a)
#     return goal_value
# def goal_function(a,b,strain):
#     # 接受理论模型计算的应力值、试验应力值和试验应变值列表
#     Num_a = len(a)
#     Num_b = len(b)
#     if Num_a != Num_b:
#         print('两个应力列表容量不相同，报错！')
#         raise ValueError
#     w = [1,1,1,1,1]
#     total = 0
#     # total = 0
#     quart = 101
#     strain_l1 = -0.005
#     strain_l2 = -0.015
#     strain_u1 = 0.005
#     strain_u2 = 0.015
#     for i in range(quart,Num_a):
#         if (strain[i] <= strain_l2):
#             total += ((a[i]-b[i]) / b[i])**2 * w[0]
#         if ((strain[i] > strain_l2) & (strain[i] <= strain_l1)):
#             total += ((a[i]-b[i]) / b[i])**2 * w[1]
#         if ((strain[i] > strain_l1) & (strain[i] <= strain_u1)):
#             total += ((a[i]-b[i]) / b[i])**2 * w[2]
#         if ((strain[i] > strain_u1) & (strain[i] <= strain_u2)):
#             total += ((a[i]-b[i]) / b[i])**2 * w[3]
#         if (strain[i] > strain_u2):
#             total += ((a[i]-b[i]) / b[i])**2 * w[4]
#         # total = total + ((a[i]-b[i])/b[i])**2
#     goal_value = np.sqrt((total/ Num_a))
#     return goal_value

# def goal_function(a,b,strain):
#     a = np.array(a)
#     b = np.array(b)
#     strain = np.array(strain)
#     Num_a = len(a)
#     Num_b = len(b)
#     if Num_a != Num_b:
#         print('两个应力列表容量不相同,报错!!!')
#         raise ValueError
#     w = [1,1,1,1,1]
#     total = 0
#     # total = 0
#     quart = 101
#     strain_l1 = -0.005
#     strain_l2 = -0.015
#     strain_u1 = 0.005
#     strain_u2 = 0.015
#     for i in range(quart, Num_a):
#         if (strain[i] <= strain_l2):
#             total += ((a[i]-b[i]) / b[i]) ** 2 * w[0]
#         if ((strain[i] > strain_l2) and (strain[i] <= strain_l1)):
#             total += ((a[i]-b[i]) / b[i]) ** 2 * w[1]
#         if ((strain[i] > strain_l1) and (strain[i] <= strain_u1)):
#             total += ((a[i]-b[i]) / b[i]) ** 2 * w[2]
#         if ((strain[i] > strain_u1) and (strain[i] <= strain_u2)):
#             total += ((a[i]-b[i]) / b[i]) ** 2 * w[3]
#         if (strain[i] > strain_u2):
#             total += ((a[i]-b[i]) / b[i]) ** 2 * w[4]
#     goal_value = np.sqrt((total / Num_a))
#     return goal_value

# def goal_function(a,b,strain):
#     a = np.array(a)
#     b = np.array(b)
#     strain = np.array(strain)
#     Num_a = len(a)
#     Num_b = len(b)
#     if Num_a != Num_b:
#         print('两个应力列表容量不相同,报错!!!')
#         raise ValueError
#     # w = [1,1,1,1,1]
#     total = 0
#     time = 0
#     # total = 0
#     quart = 101
#     # strain_l1 = -0.005
#     # strain_l2 = -0.015
#     # strain_u1 = 0.005
#     # strain_u2 = 0.015
#     for i in range(quart,Num_a):
#         if (strain[i] - strain[i-1]) >= 0:
#             total += ((a[i]-b[i]) / b[i]) ** 2 #* w[0]
#             time += 1
#     goal_value = np.sqrt((total / time))
#     return goal_value
def goal_function(a,b,strain):
    a = np.array(a)
    b = np.array(b)
    # dp = np.array(dp)
    strain = np.array(strain)
    Num_a = len(a)
    Num_b = len(b)
    if Num_a != Num_b:
        print('两个应力列表容量不相同,报错!!!')
        raise ValueError
    # w = [1,1,1,1,1]
    total = 0
    total_b = 0
    time = 0
    # total = 0
    # quart = 102
    # quart = 42
    quart = 20
    # strain_l1 = -0.005
    # strain_l2 = -0.015
    # strain_u1 = 0.005
    # strain_u2 = 0.015
    for i in range(quart,Num_a):
        if (strain[i] - strain[i-1]) > 0:
            total += (a[i] - b[i]) ** 2
            total_b += b[i] ** 2
            # total += ((a[i]-b[i]) / b[i]) ** 2 #* w[0]
            time += 1
    goal_value = np.sqrt((total / total_b) / time)
    return goal_value
