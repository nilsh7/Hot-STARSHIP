import numpy as np
import sympy as sp
from scipy.optimize import newton
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class Grid:
    def __init__(self):
        self.nC = 0


class FrontGrid(Grid):

    def __init__(self, length, l0, maxgrowth):

        # Calculate number of necessary cells based on exponential distribution
        self.nC = int(np.ceil(np.log(2 * (1 - length / l0 * (1 - maxgrowth)) / (1 + 1 / maxgrowth)) / np.log(maxgrowth)))

        # Calculate according growth factor
        eq_gr = lambda gr: length - l0 * (1 - 0.5 * gr ** (self.nC - 1) - 0.5 * gr ** (self.nC)) / (1 - gr)
        self.growth = newton(eq_gr, maxgrowth)

        # Calculate nodes positions and control volume boundaries
        j = np.arange(self.nC)
        self.zj = np.concatenate((np.array([0]), l0 * (1 - self.growth ** j[1:]) / (1 - self.growth)))
        self.zjp12 = np.concatenate(((self.zj[:-1] + self.zj[1:]) / 2, np.array([length])))
        self.zjm12 = np.concatenate((np.array([0]), self.zjp12[:-1]))


class DeepGrid(Grid):

    def __init__(self, length, lIni, z0):

        # Calculate number of necessary cells
        self.nC = int(np.round(length / lIni))

        # Calculate necessary growth/shrink factor to fit cells inside
        nC = self.nC
        if nC % 2 != 0:
            eq_gr = lambda gr: length - lIni * (1 + 2 * (gr * (1 - gr ** ((self.nC - 1) / 2))) / (1 - gr))
        else:
            eq_gr = lambda gr: length - lIni * (1 + 2 * gr * (1 - gr**(nC/2-1))/(1-gr) + gr**(nC/2))
        self.growth = newton(eq_gr, 1.01) if length / lIni > self.nC else newton(eq_gr, 0.99)

        # Calculate nodal positions
        gr = self.growth
        j = np.arange(self.nC)
        if nC % 2 != 0:
            mid = int((nC-1)/2)
            self.zj = np.concatenate((np.array([0.5 * lIni]),
                                      lIni * (0.5 + gr*(1-gr**j[1:mid+1])/(1-gr)),
                                      length/2 + lIni * gr**nC * (gr**(-(nC+1)/2) - gr**-(j[mid+1:]+1)) / (1-1/gr)
                                      )) + z0
        else:
            mid = int(nC/2)
            self.zj = np.concatenate((np.array([0.5 * lIni]),
                                      lIni * (0.5 + gr * (1 - gr ** j[1:mid]) / (1 - gr)),
                                      lIni * (0.5 + gr * (1 - gr**(mid-1))/(1-gr) + gr**nC * (gr**(-mid) - gr**(-(j[mid:]+1)))/(1-1/gr))
                                      )) + z0

        # Calculate positions at control volume boundaries
        self.zjp12 = np.concatenate(((self.zj[:-1] + self.zj[1:]) / 2, np.array([length + z0])))
        self.zjm12 = np.concatenate((np.array([z0]), self.zjp12[:-1]))


def plotGrids(*grids):

    # Check types
    for grid in grids:
        if not issubclass(type(grid), Grid):
            raise TypeError("You must specify a value of type grid.")

    # Create figure and axes
    fig, ax = plt.subplots(1, figsize=(10, 2))

    maxz = max([grid.zjp12[-1]] for grid in grids)[0]
    minz = min([grid.zjp12[-1]] for grid in grids)[0]

    for grid in grids:

        # Detemine height based on maximum cell width
        height = maxz / 10

        # Plot rectangles
        for zj, zjp12, zjm12 in zip(grid.zj, grid.zjp12, grid.zjm12):
            rect = Rectangle(xy=(zjm12, -height / 2), width=zjp12 - zjm12, height=height, fill=False)
            ax.add_patch(rect)

        # Determine marker size and plot node locations
        size = ((grid.zjp12 - grid.zjm12) / grid.zjp12[-1] * 10) ** 2
        plt.scatter(grid.zj, np.zeros(grid.zj.shape), c='black', s=size)

    # Modify plot looks
    plt.xlim((minz - 0.0005 * grid.zjp12[-1], 1.0005 * maxz))
    # plt.ylim((-0.6*height, 0.75*height))
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


if __name__ == "__main__":

    lFront = 100
    fgrid = FrontGrid(length=lFront, l0=lFront / 500, maxgrowth=1.1)

    dgrid = DeepGrid(length=lFront*0.55, lIni=fgrid.zjp12[-1] - fgrid.zjm12[-1], z0=lFront)

    plotGrids(fgrid, dgrid)
