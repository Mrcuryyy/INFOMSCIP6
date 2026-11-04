# To answer the second research question—the dependence of the number of misclassified points
# on f —we let f take on the values 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3. The other parameters k and
# n use their default value.

import numpy as np
import matplotlib.pyplot as plt
from experimentSetup import ExperimentSetup

array = [[] for _ in range(7)]
es = ExperimentSetup()

for i in range(7):
    for j in range(30):
        set = es.generate_set(1000, 0.05*i)
        mc = es.test_knn(set, 7)
        array[i].append(mc)
        print(f"{i}, {j}: {mc}")

array = np.array(array)

avgs = []
stds = []
for i in range(7):
    avgs.append(np.mean(array[i]))
    stds.append(np.std(array[i]))

avgs = np.array(avgs)
stds = np.array(stds)

print(avgs)
print(stds)

x = np.linspace(0, 0.3, 7)

#plt.plot(x, avgs)
#plt.axis([0, 0.3, 0, 2000])
#plt.fill_between(x, avgs - stds, avgs + stds, alpha=0.4)
#plt.show()

plt.figure(figsize=(7, 5))
plt.errorbar(
    x,
    avgs,
    yerr=stds,
    fmt="-o",
    color="blue",
    ecolor="lightblue",
    elinewidth=2,
    capsize=4,
)
plt.title(r"Experiment 2: $k$-NN Misclassifications depending on the value of $f$")
plt.xlabel("$f$")
plt.ylabel("Misclassified Points (out of 10,000)")
plt.xticks(x)
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
# Save the figure to be included in LaTeX
plt.savefig("experiment2_plot.pdf", dpi=300)
print("\nPlot saved as experiment2_plot.pdf")
plt.show()

#es.show_plot(set)
