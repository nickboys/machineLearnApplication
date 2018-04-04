import numpy as np

a=np.arange(20).reshape(4,5);
np.savetxt('demo.csv',a,fmt='%d',delimiter=',')
# np.loadtxt('demo.csv',delimiter=',')
# print(np.loadtxt('demo.csv',delimiter=','))