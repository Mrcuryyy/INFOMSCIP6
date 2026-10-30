from experimentSetup import ExperimentSetup

es = ExperimentSetup()

set = es.generate_set(1000, 0)
mc = es.test_knn(set, 7)
print(f"{mc} of 10000 points were missclassified")
es.show_plot(set)


