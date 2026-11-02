# To answer the second research question—the dependence of the number of misclassified points
# on f —we let f take on the values 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3. The other parameters k and
# n use their default value.

import numpy as np
import matplotlib.pyplot as plt
from experimentSetup import ExperimentSetup

array = np.array([np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])])
es = ExperimentSetup()

for i in range(7):
    for j in range(30):
        set = es.generate_set(1000, 0.05*i)
        mc = es.test_knn(set, 7)
        np.append(array[i], mc)
        print(f"{i}, {j}")

avgs = np.array([])
stds = np.array([])
for i in range(7):
    np.append(avgs, np.mean(array[i]))
    np.append(stds, np.std(array[i]))

print(avgs)
print(stds)

x = np.linspace(0, 0.3, 7)

plt.plot(x, avgs)
plt.axis([0, 0.3, 0, 500])
plt.fill_between(x, avgs - stds, avgs + stds, alpha=0.4)
plt.show()

#es.show_plot(set)
