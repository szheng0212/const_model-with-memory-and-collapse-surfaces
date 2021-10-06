'''
Author: your name
Date: 2021-04-06 15:04:24
LastEditTime: 2021-04-11 08:05:22
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \Constitutive_Model_parameter_optimization\main.py
'''

import xlrd
##from GA import GA
from GA_Class_Definition import GA
from function_tools import str_to_float

### 2.8 主函数
def main():

    workbook = xlrd.open_workbook(r'F:\\SZheng\\material_test\\data_manipulate\\Output_HA_6_2_1-30_4.xlsx') # 打开文件
    #   (r'C:\\Users\\SZheng\\Desktop\\zzz.xlsx')                     # 打开文件
    table = workbook.sheet_by_name('Sheet1')
    strain = table.col_values(0)              # 读取数据列 应变 列表list
    stress = table.col_values(1)              # 读取数据列 应力 列表list
    str_to_float(strain)
    str_to_float(stress)
    print('**********试验数据导入成功!!!**********')


    ### 开始计算
    best_population_history = []    ## 保留每一代最佳个体编码
    best_paramvalues_history = []    ## 保留每一代最佳参数值
    best_fitness_history = []       ## 保留每一代最佳适应度值
    best_goal_value_history = []    ## 保留每一代最佳权重目标函数值
    best_origin_value_history = []  ## 保留每一代最佳权重目标函数值对应的平权值

    worst_population_histiry = []
    worst_paramvalues_history = []
    worst_fitness_history = []
    worst_goal_value_history = []
    worst_origin_value_history = []


    
    ### 以种群数目16，参数个数10，位串数5，
    ##下限值列表
    ##上限值列表
    LB = [20000,1000,100,3000,300,0.001,0,200]
    UB = [60000,20000,5000,25000,2000,10,10,600]
    
    ##创建GA类实例
    GA_model = GA(800,8,16,LB,UB,50,0.9,0.1)
    ## 初始化种群
    population = GA_model.species_origin()
    print('***********种群初始化完成！！！*******************')


    ## *******************************计算初始群体信息***********************************
    params_true_values = GA_model.true_params_solve(population)
    print('**********参数真实值计算完成！！！*****************')
    # goal_value存储当代群体的全部目标函数值
    origin_goal_value,weight_goal_value,fitness_value = GA_model.fitness_solve(params_true_values,strain,stress)
    print('**********种群适应度值计算完成！！！***************')
    
    current_best_parameters, current_best_fitness,current_best_index = GA_model.best_find(params_true_values,fitness_value)
    current_worst_parameters,current_worst_fitness,current_worst_index = GA_model.worst_find(params_true_values,fitness_value)
    
    ## 输出计算结果 并 保存相关信息
    '''
    if current_best_fitness > best_fitness:      ## 采用精英保留的方法，保证下一代适应度最佳值一定不小于当前代
        current_best_fitness = best_fitness
        best_parameters = current_best_parameters
        
        
        print('iteration is :',i,';Best parameters:',best_parameters,';Best fitness',best_fitness)  ## 输出迭代求解信息
    '''
    best_population_history.append(population[current_best_index])
    best_paramvalues_history.append(params_true_values[current_best_index])
    best_fitness_history.append(fitness_value[current_best_index])
    best_goal_value_history.append(weight_goal_value[current_best_index])
    best_origin_value_history.append(origin_goal_value[current_best_index])
    
    worst_population_histiry.append(population[current_worst_index])
    worst_paramvalues_history.append(params_true_values[current_worst_index])
    worst_fitness_history.append(fitness_value[current_worst_index])
    worst_goal_value_history.append(weight_goal_value[current_worst_index])
    worst_origin_value_history.append(origin_goal_value[current_worst_index])

    # best_fitness = current_best_fitness     ## 记录下当前最佳适应度值

    # 输出第一代计算结果
    print('************输出计算结果！！！***************')
    print('Iteration is :0','Best parameters:',best_paramvalues_history[-1])
    print('Best fitness:',best_fitness_history[-1])
    print('Best goal_value:',best_goal_value_history[-1])
    print('Best origin_goal_value:',best_origin_value_history[-1])
    print('Worst parameters:',worst_paramvalues_history[-1])
    print('Worst fitness:',worst_fitness_history[-1])
    print('Worst goal_value:',worst_goal_value_history[-1])
    print('Worst origin_goal_value:',worst_origin_value_history[-1])
    
    
    ## 迭代参数寻优
    for i in range(1,GA_model.iter_num):
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
        GA_model.selection(population,fitness_value)   ## 改变population,不改变fitness_value
        print('**********种群选择操作完成***************')
        ## 交叉
        GA_model.crossover(population) ### 改变population
        print('**********种群交叉操作完成***************')
        ## 变异
        GA_model.mutation(population)  ## 改变population
        print('**********种群变异操作完成***************')
        
        ######## 产生新的种群(子代)，采用精英保留策略，以父代最佳个体取代子代最差个体**************


        ### 计算子代所有参数的真实十进制数值
        params_true_values = GA_model.true_params_solve(population)
        print('**********参数真实值计算完成！！！*****************')
        
        ##计算适应函数数值列表    ##  不改变params_true_values和population
        origin_goal_value,weight_goal_value,fitness_value = GA_model.fitness_solve(params_true_values,strain,stress)
        print('**********种群适应度值计算完成！！！***************')
        
        ## 采用精英保留策略，判断变异后的子代最差个体，用前一代的最优取代最差值
        son_worst_parameters, son_worst_fitness, son_worst_index = GA_model.worst_find(params_true_values, fitness_value)
        son_best_parameters, son_best_fitness, son_best_index = GA_model.best_find(params_true_values, fitness_value)

        # 将父代最好的个体直接赋值给子代最差个体
        population[son_worst_index] = best_population_history[-1]

        # 更新best_population_history
        if son_best_fitness > best_fitness_history[-1]:
            best_fitness_history.append(son_best_fitness)
            best_population_history.append(population[son_best_index])
            best_paramvalues_history.append(son_best_parameters)
            best_goal_value_history.append(weight_goal_value[son_best_index])
            best_origin_value_history.append(origin_goal_value[son_best_index])
        else:
            best_fitness_history.append(best_fitness_history[-1])
            best_population_history.append(best_population_history[-1])
            best_paramvalues_history.append(best_paramvalues_history[-1])
            best_goal_value_history.append(best_goal_value_history[-1])
            best_origin_value_history.append(best_origin_value_history[-1])
        # 更新worst_population_history
        # if son_worst_fitness < worst_fitness_history[-1]:
        worst_fitness_history.append(son_worst_fitness)
        worst_population_histiry.append(population[son_worst_index])
        worst_paramvalues_history.append(son_worst_parameters)
        worst_goal_value_history.append(weight_goal_value[son_worst_index])
        worst_origin_value_history.append(weight_goal_value[son_worst_index])
        # else:
            # worst_fitness_history.append(worst_fitness_history[-1])
            # worst_population_histiry.append(worst_population_histiry[-1])
            # worst_paramvalues_history.append(worst_paramvalues_history[-1])
        
        ##  输出信息
        print('************输出计算结果！！！***************')
        print('Iteration is :',i,'Best parameters:',best_paramvalues_history[-1])
        print('Best fitness:',best_fitness_history[-1])
        print('Best goal_value:',best_goal_value_history[-1])
        print('Best origin_goal_value:',best_origin_value_history[-1])
        print('Worst parameters:',worst_paramvalues_history[-1])
        print('Worst fitness:',worst_fitness_history[-1])
        print('Worst goal_value:',worst_goal_value_history[-1])
        print('Worst origin_goal_value:',worst_origin_value_history[-1])
    GA_model.plot(best_fitness_history)
    print('Final parameters are :',best_paramvalues_history[-1])

