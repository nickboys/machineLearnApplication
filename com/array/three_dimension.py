import numpy as np

a=np.arange(100).reshape(5,10,2)
a.tofile('a.dat',sep=',',format='%d')
# print(np.loadtxt('demo1.csv',delimiter=','))