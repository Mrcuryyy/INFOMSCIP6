import numpy as np
import matplotlib.pyplot as plt
from experimentSetup import ExperimentSetup

es = ExperimentSetup()
f = 0  # fraction of outliers
k = 7  # 7-nn
n_values = [0, 50, 80, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
is_blue_classified_red = []
is_red_classified_blue = []

for inliers in n_values:
    outliers = 1000 - inliers
    repeats_b2r = []
    repeats_r2b = []
    for repeat in range(30):
        set = es.generate_set_ex3(inliers, outliers, f)
        b2r, r2b = es.test_knn_ex3(set, k)
        repeats_b2r.append(b2r)
        repeats_r2b.append(r2b)
    is_blue_classified_red.append(repeats_b2r)
    is_red_classified_blue.append(repeats_r2b)

is_blue_classified_red = np.array(is_blue_classified_red)
is_red_classified_blue = np.array(is_red_classified_blue)

avgs_b2r = []
stds_b2r = []
for i in is_blue_classified_red:
    avgs_b2r.append(np.mean(i))
    stds_b2r.append(np.std(i))

avgs_b2r = np.array(avgs_b2r)
stds_b2r = np.array(stds_b2r)

print(f"Blue to red avg: {avgs_b2r}")
print(f"Blue to red std: {stds_b2r}")

avgs_r2b = []
stds_r2b = []
for i in is_red_classified_blue:
    avgs_r2b.append(np.mean(i))
    stds_r2b.append(np.std(i))

avgs_r2b = np.array(avgs_r2b)
stds_r2b = np.array(stds_r2b)

print(f"Red to blue avg: {avgs_r2b}")
print(f"Red to blue std: {stds_r2b}")

x = np.array(n_values) / 1000

y_expected_misclassified_zero_inliers = 8.0 / 100.0 * 10000.0  # Area of triangle / total area
y_expected_misclassified_zero_outliers = 92.0 / 100.0 * 10000.0  # Area of everything but triangle / total area

plt.figure(figsize=(7, 5))

plt.errorbar(
    x,
    avgs_b2r,
    yerr=stds_b2r,
    fmt="-o",
    color="blue",
    ecolor="lightblue",
    elinewidth=2,
    capsize=4,
    label="Experimental misclassifications (blue to red)",
)

plt.errorbar(
    x,
    avgs_r2b,
    yerr=stds_r2b,
    fmt="-o",
    color="red",
    ecolor="pink",
    elinewidth=2,
    capsize=4,
    label="Experimental misclassifications (red to blue)",
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

plt.title(r"Experiment 3: $7$-NN Misclassifications depending on the relative density")
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

for n, avg_b2r, std_b2r, avg_r2b, std_r2b in zip(n_values, avgs_b2r, stds_b2r, avgs_r2b, stds_r2b):
    print(f"{n} & {avg_b2r:.2f} & {std_b2r:.2f} & {avg_r2b:.2f} & {std_r2b:.2f} \\\\")