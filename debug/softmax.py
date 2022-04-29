import numpy as np

x = np.array([0.19, 0.18, 0.18, 0.2,0.18, 0.22, 0.19, 0.18, 0.19, 0.19 ])
T = 0.1
print(np.exp(x/T)/sum(np.exp(x/T)))


y = np.array([0.20, 0.17, 0.17, 0.19,0.18, 0.26, 0.19, 0.18, 0.19, 0.18 ])
T = 0.1
print(np.exp(y/T)/sum(np.exp(y/T)))