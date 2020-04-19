# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from scipy.stats import norm
x = np.linspace(-6, 6, 1000)
f = norm().pdf
y = f(x)

plt.subplot(2,1,1)
x1 = (x+2)/4
plt.plot(x1,y)
plt.grid()






plt.subplot(2,1,2)






#enorm = np.random.normal(2, scale=1, size=100000)/4.
enorm = np.random.normal(0, scale=.5, size=100000)
plt.hist(enorm, bins=50)