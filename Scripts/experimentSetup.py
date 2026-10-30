import numpy as np
from enum import Enum
import random
import time
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier # for knn, install via terminal by running: py -m pip install scikit-learn


class color(Enum):
    RED = 0
    BLUE = 1 

class ExperimentSetup():
    Q_SIZE = 10
    # triangle points:
    T_P0 = (3, 3)
    T_P1 = (7, 3)
    T_P2 = (7, 7)
    T_AREA = 8

    def __init__(self):
        pass

    # returns list of how many values were misclassified per run
    def repeat_experiment(self, repeats= 30, n = 1000, k = 7, f = 0):
        start = time.time()
        print(f"starting experiments with n={n} and f={f}...")
        results = []
        for i in range(repeats):
            set = self.generate_set(n, f)
            knn_set = self.determine_knn(set)
            results.append(self.nr_misclassified(set, knn_set))

        end = time.time()
        print(f"Finished {repeats} repeats of the experiment in {round(((end-start)) / 60, 2) } minutes")
        return results

    # generates a set in the form of a tuple (points, colors)
    def generate_set(self, n, f):
        points = []
        colors = []
        for i in range(n):
            x = random.uniform(0, self.Q_SIZE)
            y = random.uniform(0, self.Q_SIZE)
            points.append([x, y])
            outlier = random.uniform(0, 1)<f

            if self.inside_Triangle(x, y):
                if not outlier: 
                    colors.append(0)
                else: colors.append(1)
            else:
                if not outlier: 
                    colors.append(1)
                else: colors.append(0)
        return np.array(points), colors

    def inside_Triangle(self, px, py):
        # calculation of barycentric coordinates i got from stack overflow lol (could be optimized but i doubt that will be necessary)
        s = 1/(2*self.T_AREA)*(self.T_P0[1]*self.T_P2[0] - self.T_P0[0]*self.T_P2[1] + (self.T_P2[1] - self.T_P0[1])*px + (self.T_P0[0] - self.T_P2[0])*py)
        t = 1/(2*self.T_AREA)*(self.T_P0[0]*self.T_P1[1] - self.T_P0[1]*self.T_P1[0] + (self.T_P0[1] - self.T_P1[1])*px + (self.T_P1[0] - self.T_P0[0])*py)

        return s >= 0 and t >=0 and 1-s-t >= 0

    # returns set with colors determined by knn
    def determine_knn(self, set, k=7):
        points, colors = set
        knn = KNeighborsClassifier(n_neighbors=k+1)
        knn_colors = []

        # probably not the most elegant way to do this, but knn library is very specific in how it works
        for point in points:
            knn.fit(points, colors)
            _, nbs_indices = knn.kneighbors([point])
            nb_color = 0
            # find how many are colored in og set
            for i in range(1, k+1):
                nb_color += colors[nbs_indices[0,i]]
            # color based on color of most neighbours
            if nb_color > k/2:
                knn_colors.append(1)
            else: 
                knn_colors.append(0)

        return points, knn_colors

    # returns how many 
    def nr_misclassified(self, set, knn_set):
        amount = 0
        # the points are gonna be the same so we only need to look at the colors
        _, colors = set
        _, colors_knn = knn_set
        for i in range(len(colors)):
            if colors[i] is colors_knn[i]:
                continue
            else: amount += 1
        return amount

    def show_plot(self, set):
        points, colors = set
        plt.scatter(points[:,0], points[:,1], c = colors)
        plt.show()


es = ExperimentSetup()
'''
set = es.generate_set(1000, 0)
set_knn = es.determine_knn(set, 7)
es.show_plot(set_knn)
print(es.nr_misclassified(set, set_knn))
'''

print(es.repeat_experiment())
