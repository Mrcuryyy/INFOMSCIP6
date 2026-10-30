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
        for _ in range(repeats):
            set = self.generate_set(n, f)
            results.append(self.test_knn(set))

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
                    colors.append(1)
                else: colors.append(0)
            else:
                if not outlier: 
                    colors.append(0)
                else: colors.append(1)
        return np.array(points), colors

    # returns if a point is inside the traingle t
    def inside_Triangle(self, px, py):
        # calculation of barycentric coordinates i got from stack overflow lol 
        #s = 1/(2*self.T_AREA)*(self.T_P0[1]*self.T_P2[0] - self.T_P0[0]*self.T_P2[1] + (self.T_P2[1] - self.T_P0[1])*px + (self.T_P0[0] - self.T_P2[0])*py)
        #t = 1/(2*self.T_AREA)*(self.T_P0[0]*self.T_P1[1] - self.T_P0[1]*self.T_P1[0] + (self.T_P0[1] - self.T_P1[1])*px + (self.T_P1[0] - self.T_P0[0])*py)

        s = 1/(16)*((4)*px + (-4)*py)
        t = 1/(16)*(-12 + 4*py)

        return s >= 0 and t >=0 and 1-s-t >= 0


    # returns set with colors determined by knn
    def test_knn(self, set, k=7, test_amount = 10000):
        points, colors = set
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(points, colors)
        counter_mis = 0

        new_points = []
        new_colors = []
        for i in range(test_amount):
            x = random.uniform(0, self.Q_SIZE)
            y = random.uniform(0, self.Q_SIZE)
            color = knn.predict([(x, y)])
            c = color[0]
            in_t = self.inside_Triangle(x, y) #ground truth
            if (in_t and c == 0) or (not in_t and c ==1) :
                counter_mis += 1

            new_points.append([x, y])
            new_colors.append(c)

        #self.show_plot((np.array(new_points), new_colors))
        return counter_mis

    # simple function to show the plots for debugging purposes
    def show_plot(self, set):
        points, colors = set
        plt.scatter(points[:,0], points[:,1], c = colors)
        plt.show()

