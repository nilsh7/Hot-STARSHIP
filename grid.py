import numpy as np
import sympy as sp
from scipy.optimize import newton
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class Grid:
    def __init__(self):
        self.nC = 0

    def plotGrid(self):

        # Create figure and axes
        fig, ax = plt.subplots(1, figsize=(10, 2))

        # Detemine height based on maximum cell width
        height = self.xjp12[-1]/10

        # Plot rectangles
        for xj, xjp12, xjm12 in zip(self.xj, self.xjp12, self.xjm12):
            rect = Rectangle(xy=(xjm12, -height/2), width=xjp12-xjm12, height=height, fill=False)
            ax.add_patch(rect)

        # Determine marker size and plot node locations
        size = ((self.xjp12-self.xjm12)/self.xjp12[-1] * 10)**2
        plt.scatter(self.xj, np.zeros(self.xj.shape), c='black', s=size)

        # Modify plot looks
        plt.xlim((0-0.05*self.xjp12[-1], 1.05*self.xjp12[-1]))
        #plt.ylim((-0.6*height, 0.75*height))
        plt.xlabel('x [m]')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.yaxis.set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.axis('equal')
        fig.subplots_adjust(top=1.0)
        fig.subplots_adjust(bottom=0.3)
        fig.set_size_inches(10, 2, forward=True)
        plt.show()

class FrontGrid(Grid):
    def __init__(self, length, l0, maxgrowth):
        # Calculate number of necessary cells based on exponential distribution
        self.nC = np.ceil(np.log(2*(1-length/l0*(1-maxgrowth))/(1+1/maxgrowth))/np.log(maxgrowth))

        # Calculate according growth factor
        eq_gr = lambda gr : length - l0 * (1 - 0.5*gr**(self.nC-1) - 0.5*gr**(self.nC))/(1-gr)
        self.growth = newton(eq_gr, maxgrowth)

        # Calculate nodes positions and control volume boundaries
        exponents = np.arange(self.nC)
        self.xj = np.concatenate((np.array([0]), l0 * (1-self.growth**exponents[1:])/(1-self.growth)))
        self.xjp12 = np.concatenate(((self.xj[:-1]+self.xj[1:])/2, np.array([length])))
        self.xjm12 = np.concatenate((np.array([0]), self.xjp12[:-1]))

        aaa = 1

class DeepGrid(Grid):
    def __init__(self):
        pass

if __name__ == "__main__":

    grid = FrontGrid(length=10, l0=10/500, maxgrowth=1.1)

    grid.plotGrid()

    aaa = 1