import numpy as np
import matplotlib.pyplot as plt
from experimentSetup import ExperimentSetup

experiment_setup = ExperimentSetup()

# different sizes of the training set
n_values = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

# how many times to run the expriment for each n
repeats = 20

# how many k near neighbors
k = 7

# outlier fraction. probability that a generated training point has its color flipped
f = 0

means = []
standard_deviations_errors = []

print("Starting Experiment 1")

for n in n_values:
    results = experiment_setup.repeat_experiment(repeats, n, k, f)

    mean_error = np.mean(results)
    # ddof = 1 : divides by n-1 instead of n to do the standard deviation
    standard_deviation_error = np.std(results, ddof=1)

    print(
        f"n = {n:4d} | Mean Misclassified: {mean_error:6.2f} | Std Dev: {standard_deviation_error:5.2f}"
    )

    means.append(mean_error)
    standard_deviations_errors.append(standard_deviation_error)


# 4. Print table ready for LaTeX
print("\n--- LaTeX Table ---")
print(r"\begin{table}[htb]")
print(r"\centering")
print(r"\begin{tabular}{|r|r|r|}")
print(r"\hline")
print(r"$n$ & Mean Misclassified & Std. Dev. \\ \hline")
for n, m, s in zip(n_values, means, standard_deviations_errors):
    print(f"{n} & {m:.2f} & {s:.2f} \\\\")
print(r"\hline")
print(r"\end{tabular}")
print(
    r"\caption{Experiment 1: Mean misclassifications and standard deviations over 20 repeats for varying $n$ ($k=7, f=0$).}"
)
print(r"\label{tab:exp1}")
print(r"\end{table}")
# 5. Plot the figure required for Section 3.1
plt.figure(figsize=(8, 5))
plt.errorbar(
    n_values,
    means,
    yerr=standard_deviations_errors,
    fmt="-o",
    color="blue",
    ecolor="lightblue",
    elinewidth=2,
    capsize=4,
)
plt.title(r"Experiment 1: $k$-NN Misclassifications vs. Density ($n$)")
plt.xlabel("Sample Size $n$ (Density)")
plt.ylabel("Misclassified Points (out of 10,000)")
plt.xticks(n_values)
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
# Save the figure to be included in LaTeX
plt.savefig("experiment1_plot.png", dpi=300)
print("\nPlot saved as experiment1_plot.png")
plt.show()
