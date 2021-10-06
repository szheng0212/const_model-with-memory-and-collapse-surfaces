'''
Author: your name
Date: 2021-10-03 15:15:13
LastEditTime: 2021-10-03 15:42:32
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \Constitutive_Model_parameter_optimization_version0908\Heaviside.py
'''
from numpy import sign

# 编写Heaviside步函数
def Heaviside(location, radius, point):
    sign = (point - location)^2 - radius^2
    if sign < 0:
        # 在超球面内
        return 0
    
    
    if sign == 0:
        # 在超球面上
        return 1
    
    
    else:
        # 在超球面外
        return 2

# 编写Mc bracket函数
def Mac_bracket(peps_n1, kx_n, q_n):
    sign = abs(peps_n1 - kx_n) - q_n
    if sign <= 0:
        return 0
    else:
        return sign

