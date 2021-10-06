'''
Author: your name
Date: 2021-04-06 18:45:09
LastEditTime: 2021-04-15 15:20:25
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \Constitutive_Model_parameter_optimization\fitnessvalue.py
'''
import numpy as np
def sum_exp2(l):
    sum_exp = 0
    for i in range(len(l)):
        sum_exp = sum_exp + l[i]**2
    return sum_exp

def goal_function(a,b):
    Num_a = len(a)
    Num_b = len(b)

    if Num_a != Num_b:
        print('两个应力列表容量不相同，报错！')
        raise ValueError
    else:
        score_value = (np.sqrt(sum([(x-y)**2.0 for x,y in zip(a,b)])/Num_a))/np.sqrt(sum_exp2(b)/Num_a) ## 考虑会不会由于数据量太大造成大数吃小数的结果？？
        ###fit_value = 1.0/values       # 取理论应力值和试验应力值的均方根倒数作为适应度值，则问题转化为求适应度最大值问题
    return score_value
