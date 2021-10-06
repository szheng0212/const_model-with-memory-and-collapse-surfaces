# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 11:09:42 2021

@author: SZheng
"""

from function_tools import str_to_float
import xlrd
# from modified_Chaboche_Model import ChabocheModel1D
from np_ChabocheModel import ChabocheModel1D
import matplotlib.pyplot as plt
from new_total_goal_function import goal_function
# from fenduan_goal_function import goal_function

# workbook = xlrd.open_workbook(r'C:\\Users\\SZheng\\Desktop\\HA6-1.xlsx')                     # 打开文件
# workbook = xlrd.open_workbook(r'F:\\SZheng\\material_test\\data_manipulate\\Output_HA_6_2_1-30_4.xlsx')
# workbook = xlrd.open_workbook(r'C:\\Users\\SZheng\\Desktop\\第二次.xlsx')
workbook = xlrd.open_workbook(r'C:\\Users\\SZheng\\Desktop\\发郑帅.xlsx')

### wb = Workbook()
### wb = load_workbook("temp.xlsx")
##xlsx_path = ''                                            # 手动设置试验数据文件路径
#   (r'C:\\Users\\SZheng\\Desktop\\zzz.xlsx')                     # 打开文件
# table = workbook.sheet_by_name('Sheet1')



# table = workbook.sheet_by_name('简化的')
# strain = table.col_values(2)              # 读取数据列 应变 列表list
# stress = table.col_values(1)

table = workbook.sheet_by_name('Sheet2')
strain = table.col_values(0)              # 读取数据列 应变 列表list
stress = table.col_values(1)


# strain = table.col_values(0)              # 读取数据列 应变 列表list
# stress = table.col_values(1)              # 读取数据列 应力 列表list
# strain = table.col_values(9)              # 读取数据列 应变 列表list
# stress = table.col_values(10)              # 读取数据列 应力 列表list
str_to_float(strain)
str_to_float(stress)

# l = [159650.2873,211.5112796,35366.57845,26638.97861,5627.957943,12845.59044,193.2193691,0.455503898,715.3260377]
# l = [161368.8100517369,162.63957275349878,33421.1191378776,20081.40953198388,\
#       4173.006222730849,3367.927902152922,131.13892663853323,0.4793839258040674,790.232362968791]
# l = [166951.9586104952,197.40577450349284,33796.78134611258,9859.702930167132,\
#       1737.125622869132,1004.3497127053382,2193.5531554729037,3493.0224352096893,\
#           310.4575256896264,44.571585246644254,221.92766373411536,0.35513911737357845,\
#               853.8919962806666,4.992251388789548]
# l = [161289.5596,29234.8759,19208.72517,14644.09794,4814.534487,2679.213218,996.850869,785.9828815,249.444055,223.6737191,183.3286126,8.482912524,0.215950218,0]
# l = [170938.0826,297.1926948,204783.2535,37613.99995,4070.532389,3226.869609,25988.2984,19339.66955,260.0710488,0,0,0,1.112862695,607.2629044,0]
# l = [160108.8143,267.6875951,22580.7596,14631.28532,9285.707746,441.9577999,15108.47579,4751.894714,344.2780917,0,0,0,1.503909592,585.0625849,0]
# l = [179280.3471,256.5405813,42000,23081.64795,5738.437975,0,13190.00262,359.5905872,8.274257921,0,0.648575448,491.7959135,0,0,0]
# l = [179280.3471,256.5405813,42000,23081.64795,5738.437975,0,13190.00262,459.5905872,8.274257921,0,0.648575448,491.7959135,0,0,0]
# l = [179280.3471,256.5405813,42000,23081.64795,5738.437975,0,13190.00262,359.5905872,8.274257921,0,0.648575448,491.7959135,0,0,0]
# l = [160000,215.79,15433.65,10072.61,251.41,0,990.72,993.92,0.0061,0,0.2586,849.79,0,0,0]
# l = [172659.6095,325.7394178,24434.97127,6193.505472,5898.001574,3922.211096,27054.24219,5268.161076,1371.738264,\
#       0,0.550876189,677.0588656,0,0,0]
# l = [163237.5367,210.0559426,50746.68002,21440.57411,7811.504184,1614.847007,21231.28055,1511.52135,96.98733996,0,0.206747252,\
#       889.213361,294.6644732,173.8984813,0]
# l = [171514.1025,210.61966,51060.74434,18000.16689,3763.089908,2901.551849,17340.19026,3795.942589,25.73587965,0,0.357730253,691.7426984,24.61111508,\
#      160.0528336,0]
# l = [160641.1749,319.7228906,20824.47131,6727.678039,7716.498105,1734.972234,28026.73152,827.8330115,15.81861097,0,0.191609565,868.2188685,113.3669981,19.27549293,0]
# l = [160000,210.0559426,50746.68002,21440.57411,7811.504184,3000,21231.28055,1511.52135,96.98733996,0,0.2,\
#       889.213361,200,100.8984813,0]
# l = [160000,210.0559426,50746.68002,21440.57411,7811.504184,3000,21231.28055,1511.52135,96.98733996,0,0.2,\
#        889.213361,100,100.8984813,0]
# l = [170000,210.0559426,40746.68002,21440.57411,7811.504184,2500,931.28055,500.52135,50.98733996,0,0.15,\
#        889.213361,100,100.8984813,0]
# l = [160000,210.0559426,40746.68002,21440.57411,7811.504184,1200,931.28055,400.52135,30.98733996,0,0.25,\
        # 889.213361,0,0,0]
# l = [165345.3496,266.7440097,24436.64974,20813.03674,7266.324297,1258.706921,1479.918461,616.9793291,8.975838638,0,0.219550342,947.1956703,0,0,0]
# l = [160712,190.872,24231.3,14288.1,9576.35,737.585,6032.6,566.115,3.68052,0,0.356554,721.682,0,0,0]
# l = [179280.3471,256.5405813,30000,10081.64795,238.437975,0,13190.00262,1059.5905872,8.274257921,0,0.348575448,700.7959135,0,0,0]
# l = [160000.3471,256.5405813,18000,8081.64795,1208.437975,0,170.00262,69.5905872,6.274257921,0,0.308575448,750.7959135,0,0,0]

# l = [161449,288.875,475184,26241.4,4760.09,0,9983.64,745.692,3.56809,0,0.629734,593.901,0,0,0]

# l = [160506,268.853,86628.6,29991.7,3941.35,0,917.129,368.33,5.79637,0,0.228148,998.704,0,0,0]
# l = [192000,150,180000,25000,2000,0,1500,170,10,0,3,150,0,0,0]
# l = [192000,150,180000,18000,2000,0,2000,100,10,0,3,200,0,0,0]
# l = [180000,150,  250000,25000,3500,0,  3000,150,0,0,  0.4,550,0,0,0]
# l = [180000,150,  80000,20000,3500,0,  600,150,10,0,  1,450,0,0,0]
# l = [180000,150,  70000,15000,3500,0,  600,140,10,0,  1,400,0,0,0]

# l = [210000,290,  22250,3000,2000,0,  467,100,0,0,  88.6,100,0,0,0]
l = [190000,190,  22250,10000,3000,0,  567,200,0,0,  10,30,0,0,0]







testmodel = ChabocheModel1D(*l)
# theory_stress = testmodel.total_solve(strain)
theory_stress , da1,da2,da3,da4,da,a1,a2,a3,a4,a,dR1,dR2,dR3,dR,\
    R1,R2,R3,R,dpeps,peps,dp,p = testmodel.total_solve(strain)
# print (theory_stress)
# origin_value,orin_goal_first,orin_goal_mid,orin_goal_last = goal_function(theory_stress,stress,strain)
# origin_value = goal_function(theory_stress,stress)
origin_value = goal_function(theory_stress,stress,strain)
# print(origin_value,orin_goal_first,orin_goal_mid,orin_goal_last)
print(origin_value)



plt.figure()
plt.plot(strain,a1)
plt.xlabel("total strain",size = 10)
plt.ylabel('backstress a1',size = 10)
plt.title('backstress a1 evolution',size = 20)
plt.show()

plt.figure()
plt.plot(strain,a2)
plt.xlabel('total strain',size = 10)
plt.ylabel('backstress a2',size = 10)
plt.title('backstress a2 evolution',size = 20)
plt.show()

plt.figure()
plt.plot(strain,a3)
plt.xlabel('total strain',size = 10)
plt.ylabel('backstress a3',size = 10)
plt.title('backstress a3 evolution',size = 20)
plt.show()

plt.figure()
plt.plot(strain,a)
plt.xlabel('total strain',size = 10)
plt.ylabel('backstress a',size = 10)
plt.title('backstress a evolution',size = 20)
plt.show()

plt.figure()
plt.plot(strain,a1)
plt.plot(strain,a2)
plt.plot(strain,a3)
plt.plot(strain,a)
plt.xlabel('total strain',size = 10)
plt.ylabel('backstress a1,a2,a3,a',size = 10)
plt.title('backstress a1,a2,a3,a evolution',size = 20)
plt.show()

plt.figure()
plt.plot(peps,a1)
plt.xlabel('plastic strain',size = 10)
plt.ylabel('backstress a1',size = 10)
plt.title('backstress a1 evolution',size = 20)
plt.show()

plt.figure()
plt.plot(peps,a2)
plt.xlabel('plastic strain',size = 10)
plt.ylabel('backstress a2',size = 10)
plt.title('backstress a2 evolution',size = 20)
plt.show()

plt.figure()
plt.plot(peps,a3)
plt.xlabel('plastic strain',size = 10)
plt.ylabel('backstress a3',size = 10)
plt.title('backstress a3 evolution',size = 20)
plt.show()

plt.figure()
plt.plot(peps,a)
plt.xlabel('plastic strain',size = 10)
plt.ylabel('backstress a',size = 10)
plt.title('backstress a evolution',size = 20)
plt.show()

plt.figure()
plt.plot(peps,a1)
plt.plot(peps,a2)
plt.plot(peps,a3)
plt.plot(peps,a)
plt.xlabel('plastic strain',size = 10)
plt.ylabel('backstress a1,a2,a3,a',size = 10)
plt.title('backstress a1,a2,a3,a evolution',size = 20)
plt.show()

plt.figure()
plt.plot(strain,R1)
plt.xlabel('total strain',size = 10)
plt.ylabel('Elastic region value R1',size = 10)
plt.title('Elastic region value R1 evolution',size = 20)
plt.show()

plt.figure()
plt.plot(p,R1)
plt.xlabel('accumulation equivalent plastic strain',size = 10)
plt.ylabel('Elastic region value R1',size = 10)
plt.title('Elastic region value R1 evolution',size = 20)
plt.show()

plt.figure()
plt.plot(peps,R1)
plt.xlabel('plastic strain',size = 10)
plt.ylabel('Elastic region value R1',size = 10)
plt.title('Elastic region value R1 evolution',size = 20)
plt.show()

plt.figure()
plt.plot(strain,a1)
plt.plot(strain,a2)
plt.plot(strain,a3)
plt.plot(strain,a)
# plt.plot(strain,R1)
plt.plot(strain,stress)
plt.xlabel('total strain',size = 10)
plt.ylabel('backstress a1,a2,a3,a, and stress',size = 10)
plt.title('backstress a1,a2,a3,a and experiment stress evolution',size = 20)
plt.show()

plt.figure()
plt.plot(strain,a1)
plt.plot(strain,a2)
plt.plot(strain,a3)
plt.plot(strain,a)
# plt.plot(strain,R1)
plt.plot(strain,theory_stress)
plt.xlabel('total strain',size = 10)
plt.ylabel('backstress,a1,a2,a3,a, and theory stress',size = 10)
plt.title('backstress a1,a2,a3,a and theory stress evolution',size = 20)
plt.show()

plt.figure()
plt.plot(peps,a1)
plt.plot(peps,a2)
plt.plot(peps,a3)
plt.plot(peps,a)
# plt.plot(strain,R1)
plt.plot(peps,stress)
plt.xlabel('plastic strain',size = 10)
plt.ylabel('backstress a1,a2,a3,a, and experiment stress',size = 10)
plt.title('backstress a1,a2,a3,a and experiment stress evolution',size = 20)
plt.show()

plt.figure()
plt.plot(peps,a1)
plt.plot(peps,a2)
plt.plot(peps,a3)
plt.plot(peps,a)
# plt.plot(strain,R1)
plt.plot(peps,theory_stress)
plt.xlabel('plastic strain',size = 10)
plt.ylabel('backstress a1,a2,a3,a, and theory stress',size = 10)
plt.title('backstress a1,a2,a3,a, and theory stress evolution',size = 20)
plt.show()

plt.figure()
plt.plot(strain,stress)
plt.xlabel('total strain',size = 10)
plt.ylabel('experiment stress',size = 10)
plt.title('experiment stress vs total strain curve',size = 20)
plt.show()
plt.figure()
plt.plot(strain,theory_stress)
plt.xlabel('total strain',size = 10)
plt.ylabel('theory stress',size = 10)
plt.title('theory stress vs total strain curve',size = 20)
plt.show()
plt.figure()
plt.plot(strain,theory_stress)
plt.plot(strain,stress)
plt.xlabel('total strain',size = 10)
plt.ylabel('stress',size = 10)
plt.title('theory stress experiment stress vs total strain curve',size = 20)
plt.show()
