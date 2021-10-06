# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 11:58:59 2021

@author: SZheng
"""

import xlrd
import numpy as np
##from GA import GA
# from GA_Class_Definition import GA
# from modified_GA_v420 import GA
from np_GA_class import GA
from function_tools import str_to_float


workbook = xlrd.open_workbook(r'C:\\Users\\SZheng\\Desktop\\发郑帅.xlsx')
table = workbook.sheet_by_name('Sheet2')
strain = table.col_values(0)              # 读取数据列 应变 列表list
stress = table.col_values(1)

# workbook = xlrd.open_workbook(r'F:\\SZheng\\material_test\\data_manipulate\\Output_HA_6_2_1-30_4.xlsx') # 打开文件
#   (r'C:\\Users\\SZheng\\Desktop\\zzz.xlsx')                     # 打开文件
# table = workbook.sheet_by_name('Sheet1')
# strain = table.col_values(0)              # 读取数据列 应变 列表list
# stress = table.col_values(1)              # 读取数据列 应力 列表list
# strain = table.col_values(9)              # 读取数据列 应变 列表list
# stress = table.col_values(10)              # 读取数据列 应力 列表list
str_to_float(strain)
str_to_float(stress)
strain = np.array(strain)
stress = np.array(stress)

print('**********试验数据导入成功!!!**********')


### 开始计算
best_population_history = []    ## 保留每一代最佳个体编码
best_paramvalues_history = []    ## 保留每一代最佳参数值
best_fitness_history = []       ## 保留每一代最佳适应度值
# best_goal_value_history = []    ## 保留每一代最佳权重目标函数值
best_origin_value_history = []  ## 保留每一代最佳权重目标函数值对应的平权值

# best_first_goal_history = []
# best_mid_goal_history = []
# best_last_goal_history = []


# '''
worst_population_histiry = []
worst_paramvalues_history = []
worst_fitness_history = []
# worst_goal_value_history = []
worst_origin_value_history = []

# worst_first_goal_history = []
# worst_mid_goal_history = []
# worst_last_goal_history = []
# '''


    ### 以种群数目16，参数个数10，位串数5
    ##下限值列表
    ##上限值列表
# LB = [150000,150,10000,1000,0,0,0,0,200]
# UB = [200000,320,50000,30000,6000,15000,1000,5,900]

# LB = [150000,150,20000,1000,500,0,1000,200,300,0,50,0,100,0]
# UB = [200000,320,50000,20000,6000,2000,30000,20000,5000,10,1000,5,1000,0]
# LB = [150000,150,20000,1000,500,0,1000,200,0,0,50,0,100,0]
# UB = [200000,320,50000,20000,6000,1000,30000,20000,200,10,1000,5,1000,0]
# LB = [160000,110,200000,20000,500,100,20000,2000,100,0,0,0,0,100,0]
# UB = [210000,300,600000,40000,6000,4000,50000,20000,600,0,0,0,5,900,0]
# b的值或者说
# LB = [160000,120,20000,5000, 1000, 50,  10000, 1000, 0,  0, 0,200,0,0,0]
# UB = [200000,350,60000,30000,10000,4000,30000,10000,1000,0, 5,900,0,0,0]
# LB = [160000,120,20000,5000, 1000, 10,  10000, 500, 0,   0, 0,200,0,  0,  0]
# UB = [200000,350,60000,30000,10000,4000,30000,10000,500, 0, 1,900,300,200,0]
# LB = [160000,150,20000,10000, 2000, 10,  500, 100, 0, 0, 0,300, 0,0,0]
# UB = [200000,300,60000,30000,10000,3000,2000,1000,10, 0, 1,1000, 0,0,0]
# LB = np.array([160000,150,30000,10000, 2000, 10,  500, 100, 0, 0, 0,300, 0,0,0])
# UB = np.array([190000,300,60000,30000,10000,2000,5000,1000,100, 0, 5,1000, 300,200,0])
# LB = np.array([160000,100,20000,10000, 2000, 10,  500, 50, 0,   0, 0,300,  0,  50,  0])
# UB = np.array([210000,300,60000,30000,10000,5000,5000,1000,100, 0, 2,1000, 500,300, 0])

# LB = np.array([160000,150,5000,2000, 100, 0,  100, 10, 0,   0, 0,300,  0,  0,  0])
# UB = np.array([210000,300,500000,30000,6000,0,1000,500,10, 0, 5,1000, 0,0, 0])

# LB = np.array([180000,100,30000,10000,1500,0,200,  100, 0, 0, 0.5, 100, 0,   100,0])
# UB = np.array([200000,200,80000,30000,4000,0,1500, 200,10, 0, 1.5, 300, 0.5, 300,0])
LB = np.array([170000,180,20000,3000,1000,0,300,  50, 0, 0, 0, 0, 0,   0,0])
UB = np.array([210000,340,30000,12000,4000,0,1000, 300,10, 0, 10, 100, 0, 0,0])
    ##创建GA类实例
GA_model = GA(700,15,30,LB,UB,20,0.9,0.1)
## 初始化种群
population = GA_model.species_origin()
print('***********种群初始化完成!!!*******************')


    ## *******************************计算初始群体信息***********************************
params_true_values = GA_model.true_params_solve(population)
print('**********参数真实值计算完成!!!*****************')
    # goal_value存储当代群体的全部目标函数值
print('********正在求解理论应力值!!!***************')
# origin_goal_value,fitness_value,orin_goal_first,orin_goal_mid,orin_goal_last = GA_model.fitness_solve(params_true_values,strain,stress)
origin_goal_value, fitness_value = GA_model.fitness_solve(params_true_values,strain,stress)
print('********理论应力值计算完成!!!***************')
print('**********种群适应度值计算完成!!!***************')

current_best_parameters, current_best_fitness, current_best_index,\
    current_worst_parameters, current_worst_fitness, current_worst_index = GA_model.best_worst_find(params_true_values, fitness_value)
# current_best_parameters, current_best_fitness,current_best_index = GA_model.best_find(params_true_values,fitness_value)
# current_worst_parameters,current_worst_fitness,current_worst_index = GA_model.worst_find(params_true_values,fitness_value)

    ## 输出计算结果 并 保存相关信息

best_population_history.append(population[current_best_index])
best_paramvalues_history.append(current_best_parameters)
best_fitness_history.append(current_best_fitness)
# best_first_goal_history.append(orin_goal_first[current_best_index])
# best_mid_goal_history.append(orin_goal_mid[current_best_index])
# best_last_goal_history.append(orin_goal_last[current_best_index])
best_origin_value_history.append(origin_goal_value[current_best_index])

# '''
worst_population_histiry.append(population[current_worst_index])
worst_paramvalues_history.append(params_true_values[current_worst_index])
worst_fitness_history.append(fitness_value[current_worst_index])
# worst_goal_value_history.append(weight_goal_value[current_worst_index])
worst_origin_value_history.append(origin_goal_value[current_worst_index])
# worst_first_goal_history.append(orin_goal_first[current_worst_index])
# worst_mid_goal_history.append(orin_goal_mid[current_worst_index])
# worst_last_goal_history.append(orin_goal_last[current_worst_index])
# '''

# best_fitness = current_best_fitness     ## 记录下当前最佳适应度值

    # 输出第一代计算结果
print('************输出计算结果！！！***************')
print('Iteration is :0','Best parameters:',best_paramvalues_history[-1])
print('Best fitness:',best_fitness_history[-1])
# print('Best goal_value:',best_goal_value_history[-1])
print('Best origin_goal_value:',best_origin_value_history[-1])
# print('Best first_goal:',best_first_goal_history[-1])
# print('Best mid_goal:',best_mid_goal_history[-1])
# print('Best last_goal:',best_last_goal_history[-1])

# '''
print('Worst parameters:',worst_paramvalues_history[-1])
print('Worst fitness:',worst_fitness_history[-1])
# print('Worst goal_value:',worst_goal_value_history[-1])
print('Worst origin_goal_value:',worst_origin_value_history[-1])
# print('Worst first_goal:',worst_first_goal_history[-1])
# print('Worst mid_goal:',worst_mid_goal_history[-1])
# print('Worst last_goal:',worst_last_goal_history[-1])
print('******************************************')
# '''
    ## 迭代参数寻优
for i in range(1, GA_model.iter_num):
    ### 计算个体所有参数的真实十进制值
        # params_true_values = model.true_params_solve(population)    ## 不改变population

    ##计算适应函数数值列表    ##  不改变params_true_values和population
        # fitness_value = model.fitness_solve(params_true_values,population,strain,stress)


        ## 寻找当前种群最好的参数值和最优适应度函数值，不改变params_true_values和population
        # current_parameters, current_fitness,best_index = model.best_find(params_true_values,fitness_value)
        ## 与之前的最优适应度函数值比较，如果更优秀则替换最优适应度函数值和对应的参数
        # results.append(best_fitness)    ## 保留每一代的最佳个体的适应度值
        # parameters.append(best_parameters)  ## 保留每一代的最佳个体的参数值

        ## 种群更新
        ## 选择
    # GA_model.selection(population, fitness_value)
    population, fitness_value = GA_model.selection(population, fitness_value)   ## 改变population,不改变fitness_value
    # GA_model.selection(population, fitness_value)
    print('**********种群选择操作完成***************')
        ## 交叉
    # GA_model.crossover(population, fitness_value)
    population = GA_model.crossover(population, fitness_value) ### 改变population
    print('**********种群交叉操作完成***************')
        ## 变异
    # GA_model.mutation(population)
    population = GA_model.mutation(population)  ## 改变population
    print('**********种群变异操作完成***************')

        ######## 产生新的种群(子代)，采用精英保留策略，以父代最佳个体取代子代最差个体**************


        ### 计算子代所有参数的真实十进制数值
    params_true_values = GA_model.true_params_solve(population)
    print('**********参数真实值计算完成!!!*****************')

        ##计算适应函数数值列表    ##  不改变params_true_values和population
    print('********正在求解理论应力值!!!***************')
    origin_goal_value, fitness_value = GA_model.fitness_solve(params_true_values, strain, stress)
    print('********理论应力值计算完成!!!***************')
    print('**********种群适应度值计算完成!!!***************')

        # 采用精英保留策略，判断变异后的子代最差个体，用前一代的最优取代最差值
    # son_worst_parameters, son_worst_fitness, son_worst_index = GA_model.worst_find(params_true_values, fitness_value)
    # son_best_parameters, son_best_fitness, son_best_index = GA_model.best_find(params_true_values, fitness_value)
    son_best_parameters, son_best_fitness, son_best_index,\
        son_worst_parameters, son_worst_fitness, son_worst_index = GA_model.best_worst_find(params_true_values, fitness_value)
        # 更新best_population_history
    if (son_best_fitness > best_fitness_history[-1]):
        best_fitness_history.append(son_best_fitness)
        best_population_history.append(population[son_best_index])
        best_paramvalues_history.append(son_best_parameters)
        best_origin_value_history.append(origin_goal_value[son_best_index])
        # best_first_goal_history.append(orin_goal_first[son_best_index])
        # best_mid_goal_history.append(orin_goal_mid[son_best_index])
        # best_last_goal_history.append(orin_goal_last[son_best_index])

    else:
        best_fitness_history.append(best_fitness_history[-1])
        best_population_history.append(best_population_history[-1])
        best_paramvalues_history.append(best_paramvalues_history[-1])
        # best_goal_value_history.append(best_goal_value_history[-1])
        # best_first_goal_history.append(orin_goal_first[-1])
        # best_mid_goal_history.append(orin_goal_mid[-1])
        # best_last_goal_history.append(orin_goal_last[-1])
        best_origin_value_history.append(best_origin_value_history[-1])

    # '''
    # 更新worst_population_history
    if (son_worst_fitness < worst_fitness_history[-1]):
        worst_fitness_history.append(son_worst_fitness)
        worst_population_histiry.append(population[son_worst_index])
        worst_paramvalues_history.append(son_worst_parameters)
        # worst_goal_value_history.append(weight_goal_value[son_worst_index])
        worst_origin_value_history.append(origin_goal_value[son_worst_index])
        # worst_first_goal_history.append(orin_goal_first[current_worst_index])
        # worst_mid_goal_history.append(orin_goal_mid[current_worst_index])
        # worst_last_goal_history.append(orin_goal_last[current_worst_index])
    else:
        worst_fitness_history.append(worst_fitness_history[-1])
        worst_population_histiry.append(worst_population_histiry[-1])
        worst_paramvalues_history.append(worst_paramvalues_history[-1])
        worst_origin_value_history.append(worst_origin_value_history[-1])
        # '''


    # 在完成计算并记录计算结果后，将父代最好的个体,信息,及适应度值 直接赋值给子代最差个体
    # 是将本代最好的个体直接赋值下一代还是历史上最好的个体直接赋值呢？
    # 必须保证种群中的个体与适应度列表一一对应！
    if (best_population_history[-1] not in population):
        population[son_worst_index] = best_population_history[-1]
        fitness_value[son_worst_index] = best_fitness_history[-1]



        ##  输出信息
    print('************输出计算结果!!!***************')
    print('Iteration is :',i,'Best parameters:',best_paramvalues_history[-1])
    print('Best fitness:',best_fitness_history[-1])
    # print('Best goal_value:',best_goal_value_history[-1])
    # print('Best origin_goal_value:',best_origin_value_history[-1])
    print('Best origin_goal_value:',best_origin_value_history[-1])
    # print('Best first_goal:',best_first_goal_history[-1])
    # print('Best mid_goal:',best_mid_goal_history[-1])
    # print('Best last_goal:',best_last_goal_history[-1])
    print('Worst parameters:',worst_paramvalues_history[-1])
    print('Worst fitness:',worst_fitness_history[-1])
    # print('Worst goal_value:',worst_goal_value_history[-1])
    print('Worst origin_goal_value:',worst_origin_value_history[-1])
    # print('Worst first_goal:',worst_first_goal_history[-1])
    # print('Worst mid_goal:',worst_mid_goal_history[-1])
    # print('Worst last_goal:',worst_last_goal_history[-1])
    print('******************************************')

GA_model.plot(best_fitness_history)
print('Final parameters are :',best_paramvalues_history[-1])

