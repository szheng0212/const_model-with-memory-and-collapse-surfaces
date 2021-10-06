'''
Author: your name
Date: 2021-04-08 10:33:30
LastEditTime: 2021-04-10 14:41:43
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \Constitutive_Model_parameter_optimization\test.py
'''
def Mc_bracket(a):  ##不改变数值a的值
    if a >= 0:
        back = a
    else:
        back = 0
    return back
if __name__ == '__main__':
    print(Mc_bracket(1))
    print(Mc_bracket(-1))

def str_to_float(str_l):    ## 改变列表str_l的值
    for i in range(len(str_l)-1,-1,-1):     ## 倒序遍历list所有元素，防止越界；删除列表中所有空字符''
        if '' in str_l:
            str_l.remove('')

        