
"""
Created on Tue Apr 13 09:06:49 2021

@author: SZheng
"""


import random
import math
import xlrd
import matplotlib.pyplot as plt
from ChabocheModel1D import ChabocheModel1D
from fenduan_goal_function import goal_function

## 2. GA优化算法
class GA(object):
###2.1 初始化    
    def __init__(self,population_size,gene_num,gene_length,LB,UB,iter_num,pc,pm):
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
        self.LB = LB                # 分别存储各个参数的下限值
        self.UB = UB                # 分别存储各个参数的上限值
        self.pc = pc   ##一般取值0.4~0.99
        self.pm = pm   ##一般取值0.0001~0.1
        
    def species_origin(self):
        '''
        初始化种群、染色体、基因
        input:self(object):                 定义的类参数
        output:population(list):            种群
        '''
        population = []
        ## 分别初始化两个染色体        
        for i in range(self.population_size):
            tmp1 = []  ##暂存器1，用于暂存一个染色体的全部可能二进制基因取值
            for j in range(self.gene_num):
                tmp2 = [] ##暂存器2，用于暂存一个染色体的基因的每一位二进制取值
                for l in range(self.gene_length):
                    tmp2.append(random.randint(0,1))
                tmp1.append(tmp2)
            population.append(tmp1)
        return population                   # population:list[][][] 三维列表


###2.2 计算适应度函数值
    def true_params_solve(self,population): ## 不改变  列表population的值
    ### def translation(self,population):           # 将二进制位串转化为十进制数字
        '''将染色体的二进制基因转换为十进制取值
        input:self(object):定义的类参数
        population(list[][][]):种群  依次对应 染色体（一套参数） 基因（一组位串） 位串值（0 or 1）
        output:population_decimalism(list):种群每个染色体取值的十进制数
        '''
        population_decimalism = []
        for i in range(self.population_size):            ## 遍历种群数（所有个体）
            tmp = []                                ##暂存器，用于暂存一个染色体的全部可能十进制取值
            for j in range(self.gene_num):         # 遍历每条染色体上的所有基因，即待优化参数的个数
                total = 0                          ## 暂存器，存储一条染色体上的所有基因值，即一套参数的数值
                for l in range(self.gene_length):  # 遍历每个基因的位串长度，即单个基因的编码长度
                    total = total + population[i][j][l] *\
                        (math.pow(2,self.gene_length - l - 1))      # 确保每个十进制数字都是整型
                tmp.append(total)               # tmp（list[]）: 存储了每条染色体上的所有参数基因表现型的十进制数值，为一一维列表
            population_decimalism.append(tmp)   # population_decimalism（list[][]）: 存储了一代群体中每个
                                                # 染色体（个体）上所有参数的基因表现型的十进制数值
        ### return population_decimalism            # list[][] : 返回该代群体所有参数十进制数值二维列表,此时尚未考虑上下限值

    ####def fitness(self,population):
    ### def true_params_solve(self,population):     ## 将上面的基因表现型十进制数字转换为考虑单参数上下限数值的 参数真实数值

        true_params_value = []
        ### population_decimalism = self.translation(population)## population_decimalism:list[i][j]表示第i个染色体的第j个基因位表示的十进制数值

        for i in range(self.population_size):         # 遍历每代的所有染色体（个体） （在此认为每条染色体即为一个个体）
            tmp = []             ##暂存器，用于暂存每组染色体十进制数值
            for j in range(self.gene_num):        # 遍历染色体上的每个基因 第j个基因值
                value = 0
                value = self.LB[j]+population_decimalism[i][j] * (self.UB[j]-self.LB[j])/\
                        (math.pow(2,self.gene_length) - 1)   # 根据上下限值（由参数上下限列表提供）以及基因（染色体）位串长度、
                                                                # 基因表现型十进制数值确定该参数的十进制数值             
                tmp.append(value)               # temp[]：list，存储每条染色体上的所有基因值（即一套完整参数的取值）
            true_params_value.append(tmp)     # true_params_value[][]:list[][]，二维列表，存储当前代群体中所有个体（染色体）
                                                # 的十进制 真实数值
        return true_params_value


        """
        以下为无用信息
        # rbf_SVM 的3-flod交叉验证平均值为适应度函数值
        ## 防止参数值为0
        if tmp[0] == 0.0:
            tmp[0] = 0.5
        if tmp[1] == 0.0:
            tmp[1] = 0.5
        """
        #### 根据上面求出的当前代的所有染色体的所有参数的真实值列表true_params_value[][],进行本构模型的迭代并求解各自的适应度值
        '''
        rbf_svm = svm.SVC(kernel = 'rbf', C = abs(tmp[0]), gamma = abs(tmp[1]))
        cv_scores = cross_validation.cross_val_score(rbf_svm,trainX,trainY,cv =3,scoring = 'accuracy')
        fitness.append(cv_scores.mean())
        '''

    def population_stress_solve(self,true_params_value,exp_data_strain):        ## 接受一个exp_data_strain 即试验数据 应力列表
        """
            params1: 接受一个当前代种群数量的列表，该列表内元素为当前代所有个体表现型
            计算每一组染色体对应的适应度函数值
            input:self(object):             定义的类参数
            true_params_value(list[][]):              染色体上所有参数的真实值 
            population(list):               种群
            output:fitness_value(list):     每一组染色体对应的适应度函数值
        """
        
        
        ##********************* 不要把一个列表赋值给另一个列表 **************************


        theory_stress_population = []      ## 存储该代种群内 所有个体参数 计算而来的 理论应力列表 list[][]
            ##  循环计算 每个个体（每套参数）的 理论应力列表stress[]
        for i in range(self.population_size):
            stress = []       ## 暂存器，用以存储  单条染色体（一套参数）的 理论计算 应力列表 stress[]
            
            model = ChabocheModel1D(*true_params_value[i])  ## 将 * + 列表名 作为地址，直接将该列表内的元素传递给class chabochemodel1D
            
            stress = model.total_solve(exp_data_strain)
            
            theory_stress_population.append(stress)        ## list[][]
        return theory_stress_population


    def fitness_solve(self,true_params_value,exp_data_strain,exp_data_stress):
        fitness_value = []
        origin_goal_value = []
        # weight_goal_value = []
        orin_goal_first_l = []
        orin_goal_mid_l = []
        orin_goal_last_l = []
        cycle_weight_value = [1.15, 1, 1] #每圈的重要性权重
        strain_weight_value = [1.05, 1.1, 1.15, 1.15, 1.1, 1.1, 1.15, 1.15, 1.1, 1.05] #每个应变分段的重要性权重
        
        theory_stress_population = self.population_stress_solve(true_params_value,exp_data_strain)     ## theory_stress_population[][]
        w = 1.3 #总数据的目标函数值权重
        for i in range(self.population_size):
        # 每次循环的初始化
            total = 0
            origin_value = 0
            orin_goal_first = []
            orin_goal_mid = []
            orin_goal_last = []
        
            origin_value,orin_goal_first,orin_goal_mid,orin_goal_last = goal_function(theory_stress_population[i],exp_data_stress,exp_data_strain)
            ## origin_value是平权的目标函数值，weight_values是加权目标函数值； 对于求最小值问题，可计算倒数作为适应度指标

            # 计算第i个个体的带权重适应度值，在此采用先求倒数，再分配权重的方法
            tmp = 0
            for j in range(len(orin_goal_first)):#10个元素
                    tmp = tmp + 1/(orin_goal_first[j])*cycle_weight_value[0]*strain_weight_value[j]+\
                        1/(orin_goal_mid[j])*cycle_weight_value[1]*strain_weight_value[j]+\
                            1/(orin_goal_last[j])*cycle_weight_value[2]*strain_weight_value[j]
            
            # 引入整体目标函数权重
            total = tmp + 1/origin_value * w

            fitness_value.append(total)           ###fitness_value[]:list[] 存储的即为加权过的适应度值
            origin_goal_value.append(origin_value)
            orin_goal_first_l.append(orin_goal_first)
            orin_goal_mid_l.append(orin_goal_mid)
            orin_goal_last_l.append(orin_goal_last)
        return origin_goal_value,fitness_value,orin_goal_first_l,orin_goal_mid_l,orin_goal_last_l

        ####
        ##将适应度函数值中为负数的数值排除  (在本例中不再考虑，当引入适应度值评价的 罚函数 penalty function时可以考虑引入)
        '''
        fitness_value = []
        num = len(fitness)
        for l in range(num):
            if (fitness[l] > 0):
                tmp1 = fitness[l]
            else:
                tmp1 = 0.0
            fitness_value.append(tmp1)
        return fitness_value
        '''

###2.3 选择操作
    def sum_value(self,fitness_value):
        '''
        适应度求和
        input:self(object):定义的类参数
        fitness_value(list):每组染色体对应的适应度函数值
        output:total(float):适应度函数值之和
        '''
        total = 0
        for i in range(len(fitness_value)):
            total = total + fitness_value[i]
        return total

    def cumsum(self,fitness1):      ## 使用轮盘赌方法 选择 selection operator
        '''计算适应度函数值累加列表
        input:self(object):定义的类参数
        fitness1(list):适应度函数值列表
        output:适应度函数值累加列表
        '''
        tmp = []    ## 创建fitness1的副本，防止函数改变列表参数的值！
        for j in range(len(fitness1)):
            tmp.append(fitness1[j])     #存储累加列表，防止cumsum改变fitness1的值
        
        ##计算适应度函数值累加列表
        for i in range(len(tmp)-1,-1,-1):            # range(start,stop,[step]) #     倒计数：从最后一位迭代至第一位
            total = 0.0
            j=0
            while(j<=i):
                total = total + tmp[j]
                j = j + 1
            tmp[i] = total         ## 列表直接赋值，还是会改变列表参数的值！！！
        return tmp     ## 仅仅为了测试返回情况，无意义## 此时fitness1[]经过函数运算已经发生变化
    
    def selection(self,population,fitness_value):   ## 该方法不修改fitness_value的值
        '''选择操作
        input:self(object):     定义的类参数
        population(list):       当前种群 list[][][]
        fitness_value(list):    每一个体对应的适应度函数值
        '''
# *********************考虑使用改进的轮盘赌方法****************************
    # 最佳保留选择：首先按轮盘赌选择方法执行遗传算法的选择操作，然后将当前群体中适应度最高的个体结构完整地复制到下一代群体中
        new_population = []
        new_fitness = [] ## 用于存储 个体占当代种群总适应度值的比例
        total_fitness = self.sum_value(fitness_value)       ## 适应度函数值之和,在此已经调用了一次，注意不要连续调用！
        for i in range(len(fitness_value)):
            new_fitness.append(fitness_value[i] / total_fitness)    ## new_fitness存储了各个个体的适应度百分比占比
        roulette_list = self.cumsum(new_fitness) ## 此时roulette_list是适应度累计值列表，作为轮盘赌的工具。没有改变new_fitness[]


        for pop_num in range(self.population_size):
            ms = []
            for i in range(2):
                ms.append(random.random())
            ms.sort()

            ## 记录 轮盘赌选择个体索引
            roulette_index = []

            fit_in = 0
            new_in = 0

            ## 轮盘赌方式选择个体
            while ((new_in < len(ms)) & (fit_in < self.population_size)):
                if(ms[new_in] < roulette_list[fit_in]): ## 寻找两次轮盘赌选择出的个体
                    roulette_index.append(fit_in)
                ###for j in range(len(population[0])):         ## 遍历个体中的所有参数，写入新的子代new_population，其实可以按照个体直接进行选择！
                    ###for l in range(len(population[0][0][0])):
                        ###new_population[new_in][j][l] = population[fit_in][j][l]
                    new_in = new_in + 1
                ## ms是从小到大排列的，轮盘赌的选择机制与个体的排列顺序相关！！！
                else:                   ## 未被选入交配池，进入下一个适应度的判断并选择操作
                    fit_in = fit_in + 1


            ## 判断轮盘赌随机选择的两个个体适应度值，保留大的进入下一代
            if fitness_value[roulette_index[0]] > fitness_value[roulette_index[1]]:
                Max_fitness_index = roulette_index[0]
            else:
                Max_fitness_index = roulette_index[1]
            new_population.append(population[Max_fitness_index])
        population = new_population

### 2.4 ************************交叉操作**************************
##      ************************采用所有基因位的两点交叉 或者是 均匀交叉（计算量更大）********************************
    def crossover(self,population):
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
        
        ### *****************单个参数在基因长度内的两点交叉操作********************
        for i in range(0,pop_num-1,2):
            for j in range(params_num):
                if (random.random() < self.pc):
                    tmp_1 = []
                    tmp_2 = []
                    cpoint = []
                    while(len(cpoint)<2):
                        tmp_cp_num = random.randint(0,gene_len) ## 注意 tmp_cp_num是0~位串长度
                        if tmp_cp_num not in cpoint:
                            cpoint.append(tmp_cp_num)
                    cpoint.sort()

                    tmp_1.extend(population[i][j][0:cpoint[0]])
                    tmp_1.extend(population[i+1][j][cpoint[0]:cpoint[1]])
                    tmp_1.extend(population[i][j][cpoint[1]:gene_len])

                    tmp_2.extend(population[i+1][j][0:cpoint[0]])
                    tmp_2.extend(population[i][j][cpoint[0]:cpoint[1]])
                    tmp_2.extend(population[i+1][j][cpoint[1]:gene_len])

                    population[i][j] = tmp_1
                    population[i+1][j] = tmp_2


        '''
            #******************************************单点交叉*********************************
                    cpoint = random.randint(0,self.gene_length) ## 随机选择基因中的交叉点 在0和单个基因的位串数之间随机选择一个整数
                    ###实现相邻的染色体基因取值的交叉                    
                    tmp1 = []
                    tmp2 = []
                    #将tmp1作为暂存器，暂时存放 第j个染色体 第i个取值   中的    前0到cpoint个基因，
                    #然后再把   第j+1个染色体 第i个取值 中的后面的基因，补充到tem1后面
                    tmp1.extend(population[i][j][0:cpoint])     ## tmp1和tmp2 存储了 某个参数基因上面的 位串信息（序列）
                    tmp1.extend(population[i+1][j][cpoint:self.gene_length])
                    #将tmp2作为暂存器，暂时存放第j个染色体第i个取值中的前0到cpoint个基因，
                    #然后再把第j个染色体第i个取值中的后面的基因，补充到tem2后面
                    tmp2.extend(population[i+1][j][0:cpoint])
                    tmp2.extend(population[i][j][cpoint:self.gene_length])
                    #将交叉后的染色体取值放入新的种群中
                    population[i][j] = tmp1     ## 将交叉后的位串赋值给 参数 ，形成新的子代
                    population[i+1][j] = tmp2 ## 将列表赋值给population后，改变tmp1的值后population未发生改变！！！！
        #   *********************************************************************************
        '''

### 2.5 变异操作
    def mutation(self,population):
        '''变异操作
        input:self(object):         定义的类参数
        population(list):           当前种群    list[][][]
        '''
        # ********************* 全位串均匀变异 ***********************************
        paras_num = self.gene_num        # 染色体上的 基因数（参数数目）
        Gene_len = self.gene_length    # 基因长度
        pop_num =self.population_size
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



### 2.6 找出当前种群中最好的适应度和对应的参数值
    def best_find(self,true_params_value,fitness_value):
        '''找出最好的适应度和对应的参数值

        input:self(object): 定义的类参数
        true_params_value(list):   当前种群中个体的真实十进制数值 list[][]
        fitness_value:      当前适应度函数值列表
        output:[bestparameters,bestfitness]:    最优参数和最优适应度函数值
        '''
        pop_len = len(true_params_value) ## 获取当前种群数量（对应于染色体数量）  该种群已经经过解码，为各参数的真实值true_params_value[][]
        gene_num = len(true_params_value[0])
        bestparameters = []  ##用于存储当前种群最优适应度函数值对应的参数
        bestfitness = 0    ## 用于存储当前种群最优适应度函数值    初始化为0.0 float

        tmp1 = 0    ## 存储最佳个体索引值
        for i in range(pop_len):  ## 遍历种群中所有个体（染色体）
            # tmp = []      ## 暂存器 存储当前情况下的    最优适应度个体  的  十进制真实参数值
            
            # *********要保证在之前操作求解过程中 fitness_value列表的值没有发生改变！！***********
            if (fitness_value[i] > bestfitness):
                bestfitness = fitness_value[i]
                tmp1 = i
        # 思路：先找出最优适应度值的索引信息，再保存该索引对应的个体信息！
        for j in range(gene_num):     ## 遍历染色体上所有基因，即基因数目；  第i个染色体的第j个基因
            # tmp.append(true_params_value[i][j])
            bestparameters.append(true_params_value[tmp1][j])
                # bestparameters = tmp    ## bestparameters[]:list[],存储了当代种群中适应度值最高的个体的 参数的十进制真实值

        return bestparameters,bestfitness,tmp1   ## 返回给当前代的最佳适应度值的个体参数组合，及最佳适应度值

    def worst_find(self,true_params_value,fitness_value):
        ## 找出适应度最差的个体，用最佳个体代替最差个体
        pop_len = len(true_params_value)
        gene_num = len(true_params_value[0])
        worstparameters = []
        worstfitness = 50000000    ## 选一个较大的值作为初始值
        tmp1 = 0    ## 暂存器，保留最差个体的索引值
        for i in range(pop_len):
            if (fitness_value[i] < worstfitness):
                worstfitness = fitness_value[i]
                tmp1 = i
        for j in range(gene_num):
            worstparameters.append(true_params_value[tmp1][j])
        
        return worstparameters,worstfitness,tmp1

### 2.7 画出适应度函数值变化图
    def plot(self,results):

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
    
    
if __name__ == '__main__':
    ### 以种群数目16，参数个数10，位串数5，
    ##下限值列表[1,2,3,4,5,6,7,8,9,10]，
    ##上限值列表[101,102,103,104,105,106,107,108,109,110]迭代次数20为算例进行测试
    LB = [1,2,3,4,5,6,7,8,9,10]
    UB = [101,102,103,104,105,106,107,108,109,110]
    testmodel = GA(16,10,5,LB,UB,20,0.7,0.1)
    
    ## 载入试验数据
    ### xlsx_path = ''                                            # 手动设置试验数据文件路径
    xl = xlrd.open_workbook(r'C:\\Users\\SZheng\\Desktop\\zzz.xlsx')                     # 打开文件
    table = xl.sheets()[0]                                    
    test_strain = table.col_values(4)                              # 读取数据列 应变 列表list
    test_stress = table.col_values(5)                              # 读取数据列 应力 列表list
    if '' in test_strain:                                          # 删除空数据''
        test_strain.remove('')
    if '' in test_stress:                                          # 删除空数据''
        test_stress.remove('')
    print('******试验数据载入成功*******')
    print('****载入应变列表长度:',len(test_strain),'****')
    print('****载入应力列表长度:',len(test_stress),'****')
    
    
    
    ## 测试初始化函数
    print('测试初始化函数species_origin')
    pop = testmodel.species_origin()
    print(pop)

    ##测试translation函数
    pop_decimalism = testmodel.translation(pop)
    print(pop_decimalism)
    
    ## 测试true_params_solve函数
    true_params = testmodel.true_params_solve(pop)
    print(true_params)
    
    ## 测试population_stress_solve函数
    
    theory_stress = testmodel.population_stress_solve(pop,test_strain)
    print(theory_stress)
    
    ## 测试函数fitness_solve
    fitness_values = testmodel.fitness_solve(pop,test_strain,test_stress)
    print(fitness_values)
    
    ## 测试函数sum_value
    total = testmodel.sum_value(fitness_values)
    print(total)
    
    ## 测试函数cumsum
    fitness_l = testmodel.cumsum(fitness_values)
    print(fitness_l)
    
    ## 测试selection函数
    pop1 = testmodel.selection(pop,fitness_values)
    print(pop1)
    
    ## 测试函数crossover
    
    
