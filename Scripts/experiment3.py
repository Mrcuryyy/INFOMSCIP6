import numpy as np
import matplotlib.pyplot as plt
from experimentSetup import ExperimentSetup

es = ExperimentSetup()
f = 0   # fraction of outliers
k = 7   # 7-nn
n_values = [0, 50, 80, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
results = []

for inliers in n_values:
    outliers = 1000 - inliers
    repeats = []
    for repeat in range(30):
        set = es.generate_set_ex3(inliers, outliers, f)
        mc = es.test_knn(set, k)
        repeats.append(mc)
        # print(f"Inliers: {inliers}\t Outliers: {outliers}\t Misclassified: {mc}")
    results.append(repeats)


array = np.array(results)

avgs = []
stds = []
for i in results:
    avgs.append(np.mean(i))
    stds.append(np.std(i))

avgs = np.array(avgs)
stds = np.array(stds)

print(avgs)
print(stds)

x = np.array(n_values) / 1000

y_expected_misclassified_zero_inliers  = 8.0 / 100.0 * 10000.0  # Area of triangle / total area
y_expected_misclassified_zero_outliers = 92.0 / 100.0 * 10000.0 # Area of everything but triangle / total area

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
    label="Experimental misclassifications",
)

# Expected misclassifications when there are zero inliers
plt.axhline(
    y=y_expected_misclassified_zero_inliers,
    color="red",
    linestyle="--",
    linewidth=2,
    label="Expected misclassifications (0 inliers)",
)

# Expected misclassifications when there are zero outliers
plt.axhline(
    y=y_expected_misclassified_zero_outliers,
    color="green",
    linestyle="--",
    linewidth=2,
    label="Expected misclassifications (0 outliers)",
)

plt.title(r"Experiment 3: $7$-NN Misclassifications depending on the fraction of inliers")
plt.xlabel("Fraction of inliers")
plt.ylabel("Misclassified Points (out of 10,000)")
plt.yscale("log")
# plt.xticks(x)
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend()
plt.tight_layout()
# Save the figure to be included in LaTeX
plt.savefig("experiment3_plot.png", dpi=300)
print("\nPlot saved as experiment3_plot.png")
plt.show()
