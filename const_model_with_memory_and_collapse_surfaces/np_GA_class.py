# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 15:28:43 2021

@author: SZheng
"""

import numpy as np
import random
import math
import matplotlib.pyplot as plt
# from multi_parameters_chaboche import ChabocheModel1D
# from modifi_multiparameter_chaboche import ChabocheModel1D
from np_ChabocheModel import ChabocheModel1D
# from modified_Chaboche_Model import ChabocheModel1D
from new_total_goal_function import goal_function

## 2. GA优化算法
class GA(object):
###2.1 初始化
    def __init__(self, population_size, gene_num, gene_length, LB, UB, iter_num, pc, pm):
        '''初始化参数
        input:population_size(int):   种群数（或称之为每一代中的所有基因组合构成的染色体条数） 取50
            chromosome_num(int):      染色体数（或理解为基因个数），对应需要寻优的  ##参数个数 取10
            chromosome_length:        染色体上的基因长度 应该是指每个基因位的基因长度，即每个参数的位串数  在此均取为10
            max_value(float):         #####!!!!!!!!!!!        作用于二进制基因转化为染色体十进制数值
            iter_num(int):            迭代次数                    取50
            pc(float):                交叉概率阈值(0<pc<1)        取0.7
            pm(float):                变异概率阈值(0<pm<1)        取0.1
        '''
        self.population_size = population_size
        self.gene_length = gene_length
        self.gene_num = gene_num
        self.iter_num = iter_num
        self.LB = np.array(LB)                # 分别存储各个参数的下限值
        self.UB = np.array(UB)                # 分别存储各个参数的上限值
        self.pc = pc   ##一般取值0.4~0.99
        self.pm = pm   ##一般取值0.0001~0.1

    def species_origin(self):
        '''
        初始化种群、染色体、基因
        input:self(object):                 定义的类参数
        output:population(list):            种群
        '''
        # population = np.zeros([self.population_size,self.gene_num,\
        #                        self.gene_length])
        # for chorosome in population:
        #     for gene_numb in chorosome:
        #             gene_numb[...] = np.random.randint(0,2)
        population = np.random.randint(0, 2, (self.population_size, self.gene_num, self.gene_length))
        return population



###2.2 计算适应度函数值
    def true_params_solve(self, population): ## 不改变  列表population的值
    ### def translation(self,population):           # 将二进制位串转化为十进制数字
        '''将染色体的二进制基因转换为十进制取值
        input:self(object):定义的类参数
        population(list[][][]):种群  依次对应 染色体（一套参数） 基因（一组位串） 位串值（0 or 1）
        output:population_decimalism(list):种群每个染色体取值的十进制数
        '''
        multiply = np.array([math.pow(2, (self.gene_length - i - 1)) for i in range(self.gene_length)])
        population_decimalism = np.sum(population * multiply, axis = -1)

        param_range = self.UB - self.LB
        division = (math.pow(2, self.gene_length)) - 1
        true_params_value = self.LB + population_decimalism * param_range / division

        return true_params_value

    def population_stress_solve(self, true_params_value, exp_data_strain):        ## 接受一个exp_data_strain 即试验数据 应力列表
        """
            params1: 接受一个当前代种群数量的列表，该列表内元素为当前代所有个体表现型
            计算每一组染色体对应的适应度函数值
            input:self(object):             定义的类参数
            true_params_value(list[][]):              染色体上所有参数的真实值
            population(list):               种群
            output:fitness_value(list):     每一组染色体对应的适应度函数值
        """

        stress_l = []
        ##********************* 不要把一个列表赋值给另一个列表 **************************

        # theory_stress_population = []      ## 存储该代种群内 所有个体参数 计算而来的 理论应力列表 list[][]
            ##  循环计算 每个个体（每套参数）的 理论应力列表stress[]
        for i in range(self.population_size):
            stress = []       ## 暂存器，用以存储  单条染色体（一套参数）的 理论计算 应力列表 stress[]

            ChabocheModel = ChabocheModel1D(* true_params_value[i])  ## 将 * + 列表名 作为地址，直接将该列表内的元素传递给class chabochemodel1D

            stress = ChabocheModel.total_solve(exp_data_strain)

            stress_l.append(stress)        ## list[][]
        theory_stress_population = np.array(stress_l)
        return theory_stress_population

    def fitness_solve(self, true_params_value, exp_data_strain, exp_data_stress):
        fitness_value = []
        origin_goal_value = []  # 存储平权的目标函数值
        # weight_goal_value = []
        # orin_goal_first_l = []
        # orin_goal_mid_l = []
        # orin_goal_last_l = []
        # cycle_weight_value = [1.15, 1, 1] #每圈的重要性权重
        # strain_weight_value = [1.05, 1.1, 1.15, 1.15, 1.1, 1.1, 1.15, 1.15, 1.1, 1.05] #每个应变分段的重要性权重

        theory_stress_population = self.population_stress_solve(true_params_value, exp_data_strain)     ## theory_stress_population[][]
        # w = 1.3 #总数据的目标函数值权重
        for i in range(self.population_size):
        # 每次循环的初始化
            # total = 0
            # origin_value = 0
            # orin_goal_first = []
            # orin_goal_mid = []
            # orin_goal_last = []
            goal_value = goal_function(theory_stress_population[i], exp_data_stress, exp_data_strain)
            fitness = 1 / goal_value
            # origin_value,orin_goal_first,orin_goal_mid,orin_goal_last = goal_function(theory_stress_population[i],exp_data_stress,exp_data_strain)
            ## origin_value是平权的目标函数值，weight_values是加权目标函数值； 对于求最小值问题，可计算倒数作为适应度指标

            # 计算第i个个体的带权重适应度值，在此采用先求倒数，再分配权重的方法
            # tmp = 0
            # for j in range(len(orin_goal_first)):#10个元素
                    # tmp = tmp + 1/(orin_goal_first[j])*cycle_weight_value[0]*strain_weight_value[j]+\
                    #     1/(orin_goal_mid[j])*cycle_weight_value[1]*strain_weight_value[j]+\
                    #         1/(orin_goal_last[j])*cycle_weight_value[2]*strain_weight_value[j]

            # 引入整体目标函数权重
            # total = tmp + 1/origin_value * w

            # fitness_value.append(total)           ###fitness_value[]:list[] 存储的即为加权过的适应度值
            origin_goal_value.append(goal_value)
            fitness_value.append(fitness)

        origin_goal_value = np.array(origin_goal_value)
        fitness_value = np.array(fitness_value)


            # origin_goal_value.append(origin_value)
            # orin_goal_first_l.append(orin_goal_first)
            # orin_goal_mid_l.append(orin_goal_mid)
            # orin_goal_last_l.append(orin_goal_last)
        # return origin_goal_value,fitness_value,orin_goal_first_l,orin_goal_mid_l,orin_goal_last_l
        return origin_goal_value, fitness_value
        ####
        ##将适应度函数值中为负数的数值排除  (在本例中不再考虑，当引入适应度值评价的 罚函数 penalty function时可以考虑引入)
###2.3 选择操作
    def sum_value(self, fitness_value):
        '''
        适应度求和
        input:self(object):定义的类参数
        fitness_value(list):每组染色体对应的适应度函数值
        output:total(float):适应度函数值之和
        '''
        total = 0
        # fitness_value = np.asarray(fitness_value)
        total = np.sum(fitness_value)
        return total

    def cumsum(self, fitness_percent):      ## 使用轮盘赌方法 选择 selection operator
        '''计算适应度函数值累加列表
        input:self(object):定义的类参数
        fitness1(list):适应度函数值列表
        output:适应度函数值累加列表
        '''
        # fitness1 = np.array(fitness1)

        # tmp = []    ## 创建fitness1的副本，防止函数改变列表参数的值！
        # for j in range(len(fitness1)):
        #     tmp.append(fitness1[j])     #存储累加列表，防止cumsum改变fitness1的值
        tmp = fitness_percent.copy()
        for i in range(1, len(tmp)):
            tmp[i] += tmp[i-1]
        ##计算适应度函数值累加列表
        # for i in range(len(tmp)-1,-1,-1):            # range(start,stop,[step]) #     倒计数：从最后一位迭代至第一位
        #     total = 0.0
        #     j=0
        #     while(j<=i):
        #         total = total + tmp[j]
        #         j = j + 1
        #     tmp[i] = total         ## 列表直接赋值，还是会改变列表参数的值！！！
        return tmp     ## 仅仅为了测试返回情况，无意义## 此时fitness1[]经过函数运算已经发生变化

    def selection(self, population, fitness_value):   ## 该方法不修改fitness_value的值
        '''选择操作
        input:self(object):     定义的类参数
        population(list):       当前种群 list[][][]
        fitness_value(list):    每一个体对应的适应度函数值
        '''
# *********************考虑使用改进的轮盘赌方法****************************
    # 最佳保留选择：首先按轮盘赌选择方法执行遗传算法的选择操作，然后将当前群体中适应度最高的个体结构完整地复制到下一代群体中

        new_population = [] # 用以保存选择操作后新群体中各个个体
        new_fitness = [] # 用以保存选择操作后新群体中各个个体对应的适应度值
        # new_fitness = [] ## 用于存储 个体占当代种群总适应度值的比例

        total_fitness = self.sum_value(fitness_value)       ## 适应度函数值之和,在此已经调用了一次，注意不要连续调用！
        # for i in range(len(fitness_value)):
        fitness_percent = list(fitness_value / total_fitness)    ## new_fitness存储了各个个体的适应度百分比占比
        roulette_list = self.cumsum(fitness_percent) ## 此时roulette_list是适应度累计值列表，作为轮盘赌的工具。没有改变new_fitness[]


        for pop_num in range(self.population_size):
            ms = np.random.random(2)
            # ms = []
            # for i in range(2):
            # ms = np.random.random(2)
                # ms.append(random.random())
            ms.sort()

            ## 记录 轮盘赌选择个体索引
            roulette_index = []

            fit_in = 0
            new_in = 0

            ## 轮盘赌方式选择个体
            while ((new_in < len(ms)) and (fit_in < self.population_size)):
                if(ms[new_in] < roulette_list[fit_in]): ## 寻找两次轮盘赌选择出的个体
                    roulette_index.append(fit_in)
                ###for j in range(len(population[0])):         ## 遍历个体中的所有参数，写入新的子代new_population，其实可以按照个体直接进行选择！
                    ###for l in range(len(population[0][0][0])):
                        ###new_population[new_in][j][l] = population[fit_in][j][l]
                    new_in = new_in + 1
                ## ms是从小到大排列的，轮盘赌的选择机制与个体的排列顺序相关！！！
                else:                   ## 未被选入交配池，进入下一个适应度的判断并选择操作
                    fit_in = fit_in + 1
            # roulette_index = np.array(roulette_index)

            # print(roulette_index,fitness_value)
## 判断轮盘赌随机选择的两个个体适应度值，保留大的进入下一代
            if fitness_value[roulette_index[0]] > fitness_value[roulette_index[1]]:
                Max_fitness_index = roulette_index[0]
            else:
                Max_fitness_index = roulette_index[1]


            new_population.append(population[Max_fitness_index])
            new_fitness.append(fitness_value[Max_fitness_index])
            # new_population.append(population[Max_fitness_index])
            # new_fitness.append(fitness_value[Max_fitness_index])

        # 将选择操作产生的新种群的种群编码及其各个个体对应的适应度值列表
        population = np.array(new_population)
        fitness_value = np.array(new_fitness) # 返回新群体的适应度值列表
        return population, fitness_value

### 2.4 ************************交叉操作**************************
##      ************************采用所有基因位的两点交叉 或者是 均匀交叉（计算量更大）********************************
    def crossover(self, population, fitness_value):
        '''交叉操作
        input:self(object):     定义的类参数
        population(list):       当前种群    list[][][]
        '''
        params_num = self.gene_num        ## 取个体中所有参数数目
        gene_len = self.gene_length
        pop_num = self.population_size

        '''
        ### 所有位串的均匀交叉**************************************
        for i in range(0,pop_num-1,2): ## 遍历所有参数遍历群体中所有个体,考虑到最后一位发生越界，故pop_size-1
            for j in range(params_num):
            ## 在此认为在一轮交叉操作过程中，同一条染色体最多只能参与一次交叉操作！！
            ## 第i个 个体的 第j个 参数
                for l in range(gene_len):   ## 针对第i个个体第j个参数上的全部位串进行交叉判定
                    if (random.random < self.pc):
                        tmp_1 = population[i][j][l]  ##暂存第l个位串数值
                        tmp_2 = population[i+1][j][l]   ## 暂存位串值
                        population[i][j][l] = tmp_2
                        population[i+1][j][l] = tmp_1

        ### *****************************************************************
        '''
        # 在进行交叉操作前，先对群体进行排序，即按照适应度值的大小排序，保证交配的'门当户对'。
        # 先计算选择操作后种群的适应度列表
        # fitness_index数组 存储了选择操作后新产生个体的降序排列的索引值
        fitness_tmp = np.array(fitness_value).copy()
        fitness_index = np.argsort(- fitness_tmp) # 此时适应度函数已经被改变
        population_tmp = np.array(population).copy() # 拷贝population数组的副本，在计算完成后重新给population数组赋值(按地址)

        ### *****************单个参数在基因长度内的两点交叉操作********************
        for i in range(0, pop_num - 1, 2):
            for j in range(params_num):
                if (random.random() < self.pc):
                    # 判断发生交叉操作
                    tmp_1 = []
                    tmp_2 = []
                    cpoint = []
                    while(len(cpoint) < 2):
                        tmp_cp_num = random.randint(0, gene_len) ## 注意 tmp_cp_num是0~位串长度
                        if tmp_cp_num not in cpoint:
                            cpoint.append(tmp_cp_num)
                    cpoint.sort()

                    tmp_1.extend(population[fitness_index[i]][j][0 : cpoint[0]])
                    tmp_1.extend(population[fitness_index[i + 1]][j][cpoint[0] : cpoint[1]])
                    tmp_1.extend(population[fitness_index[i]][j][cpoint[1] : gene_len])

                    tmp_2.extend(population[fitness_index[i + 1]][j][0 : cpoint[0]])
                    tmp_2.extend(population[fitness_index[i]][j][cpoint[0] : cpoint[1]])
                    tmp_2.extend(population[fitness_index[i + 1]][j][cpoint[1] : gene_len])

                    population_tmp[i][j] = tmp_1
                    population_tmp[i + 1][j] = tmp_2
                else:
                    # 不发生交叉操作
                    population_tmp[i][j] = population[fitness_index[i]][j]
                    population_tmp[i + 1][j] = population[fitness_index[i + 1]][j]

        population = population_tmp

                    # tmp_1.extend(population[i][j][0:cpoint[0]])
                    # tmp_1.extend(population[i+1][j][cpoint[0]:cpoint[1]])
                    # tmp_1.extend(population[i][j][cpoint[1]:gene_len])

                    # tmp_2.extend(population[i+1][j][0:cpoint[0]])
                    # tmp_2.extend(population[i][j][cpoint[0]:cpoint[1]])
                    # tmp_2.extend(population[i+1][j][cpoint[1]:gene_len])

                    # population[i][j] = tmp_1
                    # population[i+1][j] = tmp_2


        '''
            #******************************************单点交叉*********************************
                    cpoint = random.randint(0, self.gene_length) ## 随机选择基因中的交叉点 在0和单个基因的位串数之间随机选择一个整数
                    ###实现相邻的染色体基因取值的交叉
                    tmp1 = []
                    tmp2 = []
                    #将tmp1作为暂存器，暂时存放 第j个染色体 第i个取值   中的    前0到cpoint个基因，
                    #然后再把   第j+1个染色体 第i个取值 中的后面的基因，补充到tem1后面
                    tmp1.extend(population[fitness_index[i]][j][0 : cpoint])     ## tmp1和tmp2 存储了 某个参数基因上面的 位串信息（序列）
                    tmp1.extend(population[fitness_index[i + 1]][j][cpoint : self.gene_length])
                    #将tmp2作为暂存器，暂时存放第j个染色体第i个取值中的前0到cpoint个基因，
                    #然后再把第j个染色体第i个取值中的后面的基因，补充到tem2后面
                    tmp2.extend(population[fitness_index[i+1]][j][0 : cpoint])
                    tmp2.extend(population[fitness_index[i]][j][cpoint : self.gene_length])

                    #将交叉后的染色体取值放入新的种群中
                    population_tmp[i][j] = tmp_1
                    population_tmp[i + 1][j] = tmp_2 ## 将列表赋值给population后，改变tmp1的值后population未发生改变！！！！
        population = population_tmp
        #   *********************************************************************************
        '''
        return population

    ### 2.5 变异操作
    def mutation(self, population):
        '''变异操作
        input:self(object):         定义的类参数
        population(list):           当前种群    list[][][]
        '''
        population = np.array(population)

        # ********************* 全位串均匀变异 ***********************************
        paras_num = self.gene_num        # 染色体上的 基因数（参数数目）
        Gene_len = self.gene_length    # 基因长度
        pop_num = self.population_size
        for i in range(pop_num):    ## 遍历种群所有染色体
            for j in range(paras_num):
                for l in range(Gene_len):        ### 第i个 染色体(个体)上的 第j个基因(参数)
                    if (random.random() < self.pm): # 触发变异条件
                        ##采用均匀变异的方法计算
                        if (population[i][j][l] == 1):
                            population[i][j][l] = 0
                        else:
                            population[i][j][l] = 1

        #   ***************************************************************************************

        #   ********************* 单点位串变异 ******************************************************
        """
        paras_num = self.gene_num        # 染色体上的 基因数（参数数目）
        Gene_len = self.gene_length    # 基因长度
        for i in range(self.population):    ## 遍历种群所有染色体
            for j in range(paras_num):        ### 第i个 染色体(个体)上的 第j个基因(参数)
                if (random.random() < self.pm):
                    m_point = random.randint(0,Gene_len - 1) ## m_point 记录下基因位串变异点位信息
                    ##将第mpoint个基因点随机变异，变为0或者1
                    if (population[i][j][m_point] == 1):
                        population[i][j][m_point] = 0
                    else:
                        population[i][j][m_point] = 1
        """
        return population

### 2.6 找出当前种群中最好的适应度和对应的参数值
    def best_worst_find(self, true_params_value, fitness_value):
        '''找出最好的适应度和对应的参数值

        input:self(object): 定义的类参数
        true_params_value(list):   当前种群中个体的真实十进制数值 list[][]
        fitness_value:      当前适应度函数值列表
        output:[bestparameters,bestfitness]:    最优参数和最优适应度函数值
        '''

        fitness_tmp = np.array(fitness_value).copy()
        fitness_index = np.argsort(fitness_tmp)

        bestparameters = true_params_value[fitness_index[-1]]
        bestfitness = fitness_value[fitness_index[-1]]
        best_index = fitness_index[-1]

        worstparameters = true_params_value[fitness_index[0]]
        worstfitness = fitness_value[fitness_index[0]]
        worst_index = fitness_index[0]

        return bestparameters, bestfitness, best_index, worstparameters, worstfitness, worst_index   ## 返回给当前代的最佳适应度值的个体参数组合，及最佳适应度值


### 2.7 画出适应度函数值变化图
    def plot(self, results):

        """
        画图
        Args:
        input:      self(object):       定义的类参数
        results:    results(list[]):    列表list[]，记录下每代中适应度最高的个体的适应度值

        """
        X = []
        Y = []
        for i in range(self.iter_num):
            X.append(i + 1)     # 相当于初始化第一次迭代次数为1，即不可能从 0 代开始
            Y.append(results[i])    # 记录下每次迭代的
        plt.plot(X,Y)
        plt.xlabel('Number of iteration',size = 10)
        plt.ylabel('The Best Value of Fitness',size = 10)
        plt.title('Constitutive_Model Parameters Optimization Based GA Method',size = 20)     ## 绘制的是每代迭代最佳个体的适应度值随迭代次数的关系曲线
        plt.show()


