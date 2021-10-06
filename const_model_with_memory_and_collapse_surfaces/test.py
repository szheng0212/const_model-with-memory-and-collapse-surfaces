import numpy as np
a = np.array([1,6,31,2,22])
b = np.array(a).copy()
c = np.argsort(b)
print(a,b,c)
# tmp = []
# a = []
# for i in range(10):
#     tmp.append(np.random.random())
# a = tmp
# tmp = []
# print(a,tmp)
