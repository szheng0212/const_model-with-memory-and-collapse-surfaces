import numpy as np
import random
import math
import matplotlib.pyplot as plt
class GA(object):
    ###2.1 初始化    
    def __init__(self,population_size,chromosome_num,chromosome_length,iter_num,pc,pm):
        '''初始化参数
        input:population_size(int):     种群数 取50
        chromosome_num(int):      染色体数，对应需要寻优的参数个数 取10
        chromosome_length:        染色体的基因长度 应该是指每个基因位的基因长度  在此均取为10
        max_value(float):         作用于二进制基因转化为染色体十进制数值
        iter_num(int):            迭代次数                    取50
        pc(float):                交叉概率阈值(0<pc<1)        取0.7
        pm(float):                变异概率阈值(0<pm<1)        取0.1
        '''
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.chromosome_num = chromosome_num
        self.iter_num = iter_num
        self.pc = pc   ##一般取值0.4~0.99
        self.pm = pm   ##一般取值0.0001~0.1
    def species_origin(self):
        '''初始化种群、染色体、基因
        input:self(object):                 定义的类参数
        output:population(list):            种群
        '''
        population = []
        ## 分别初始化两个染色体        
        for i in range(self.population_size):
            tmp1 = []  ##暂存器1，用于暂存一个染色体的全部可能二进制基因取值
            for j in range(self.chromosome_num):
                tmp2 = [] ##暂存器2，用于暂存一个染色体的基因的每一位二进制取值
                for l in range(self.chromosome_length):
                    tmp2.append(random.randint(0,1))
                tmp1.append(tmp2)
                print(tmp1)
            population.append(tmp1)
            print(population)
        return population

m=GA(30,10,5,15,0.7,0.1)
m.species_origin()

if __name__ == '__main__':
    m=GA(30,10,5,15,0.7,0.1)
    m.species_origin()

s