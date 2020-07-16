import numpy as np
import sympy as sp
from scipy.optimize import newton
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import warnings

p1 = lambda f: np.roll(f, shift=-1)
m1 = lambda f: np.roll(f, shift=+1)

class Grid:
    def __init__(self):
        self.nC = None
        self.growth = None
        self.zj = None
        self.zjp12 = None
        self.zjm12 = None
        self.mat = None

    def setMaterial(self, mat):
        """
sets the material property
        :param mat: material to be set
        """
        self.mat = mat

    def calculateProperties(self, calceta=False):
        """
calculates a few auxiliary properties such as control volume boundary location or constant eta coordinate
        :param length: length of grid
        :param z0: initial position of left boundary
        """
        # Calculate positions at control volume boundaries
        self.zjp12 = np.concatenate(((self.zj[:-1] + self.zj[1:]) / 2, np.array([self.length + self.z0])))
        self.zjm12 = np.concatenate((np.array([self.z0]), self.zjp12[:-1]))
        if calceta:
            self.etaj = 1 - self.zj/self.length
            self.etajp12 = 1 - self.zjp12/self.length
            self.etajm12 = 1 - self.zjm12/self.length
        self.dzjp = p1(self.zj) - self.zj
        self.dzjm = self.zj - m1(self.zj)

    def setXtoZ(self):
        """
sets moving coordinates x to stationary coordinates z
        """
        self.xj = self.zj
        self.xjp12 = self.zjp12
        self.xjm12 = self.zjm12
        self.dxjp = self.dzjp
        self.dxjm = self.dzjm


class FrontGrid(Grid):

    def __init__(self, length, l0, maxgrowth=1.1):
        """
creates grid at front of TPS
        :param length: thickness of layer
        :param l0: l0/2 is the thickness of the first volume
        :param maxgrowth: maximum allowable growth factor
        """
        self.length0 = length
        self.length = length
        self.z0 = 0

        # Calculate number of necessary cells based on exponential distribution
        self.nC = int(np.ceil(np.log(2 * (1 - length / l0 * (1 - maxgrowth)) / (1 + 1 / maxgrowth)) / np.log(maxgrowth)))

        # Calculate according growth factor
        eq_gr = lambda gr: length - l0 * (1 - 0.5 * gr ** (self.nC - 1) - 0.5 * gr ** (self.nC)) / (1 - gr)
        self.growth = newton(eq_gr, maxgrowth)

        # Calculate nodes positions and control volume boundaries
        j = np.arange(self.nC)
        self.zj = np.concatenate((np.array([0]), l0 * (1 - self.growth ** j[1:]) / (1 - self.growth)))
        self.calculateProperties(calceta=True)

        self.s = 0  # Recession amount

    def updateZ(self, delta_s):
        """
adds recession amount and recalculates nodal positions
        :param delta_s: difference in recession amount since last update
        """
        self.s += delta_s
        self.z0 += delta_s
        self.length += -delta_s
        self.zj = self.length0 - self.etaj * (self.length0 - self.s)
        self.calculateProperties(calceta=False)


class DeepGrid(Grid):

    def __init__(self, length, lIni, z0):
        """
creates grid at inside layer of TPS
        :param length: thickness of layer
        :param lIni: control volume thickness of previous layer (to avoid large changes in volume)
        :param z0: coordinate of upper edge
        """
        self.length = length
        self.z0 = z0

        # Calculate number of necessary cells
        self.nC = int(np.round(length / lIni))

        # Calculate necessary growth/shrink factor to fit cells inside
        nC = self.nC
        if nC == 1 or nC == 0:
            self.zj = np.array([z0+length/2])
            self.zjp12 = np.array([z0+length])
            self.zjm12 = np.array([z0])
            self.growth = length/lIni
            self.nC = 1
            warnings.warn("Grid at z0 = %.3f m contains only one cell and might have large volume changes. "
                          "Volume change is %.3f." % (z0, self.growth), UserWarning)
            return
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

        self.calculateProperties(calceta=True)


def plotGrids(*grids):
    """
plots the passed grids
    :param grids: variable number of grids
    """
    # Check types
    for grid in grids:
        if not issubclass(type(grid), Grid):
            raise TypeError("You must specify a value of type grid.")

    # Create figure and axes
    fig, ax = plt.subplots(1, figsize=(10, 2))

    maxz = max([grid.zjp12[-1]] for grid in grids)[0]
    minz = min([grid.zjm12[0]] for grid in grids)[0]

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
    plt.xlim((minz - 0.005 * grid.zjp12[-1], 1.005 * maxz))
    # plt.ylim((-0.6*height, 0.75*height))
    plt.xlabel('x [m]')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    #ax.axis('equal')
    fig.subplots_adjust(top=1.0)
    fig.subplots_adjust(bottom=0.3)
    fig.set_size_inches(10, 2, forward=True)
    plt.show()


def generateGrids(layers):
    """
adds grids based to a list of layers (see input.py)
    :param layers: list of layers
    :return: modified list of layers
    """
    # Generate DeepGrid for remaining grids
    for il, layer in enumerate(layers):
        if il == 0:
            grid = FrontGrid(length=layer.thickness, l0=layer.firstcell, maxgrowth=layer.maxgrowth)
        else:
            lIni = layers[il-1].grid.zjp12[-1] - layers[il-1].grid.zjm12[-1]
            z0 = layers[il-1].grid.zjp12[-1]
            grid = DeepGrid(length=layer.thickness, lIni=lIni, z0=z0)

        layers[il].grid = grid

    return layers

if __name__ == "__main__":

    lFront = 0.05
    fgrid = FrontGrid(length=lFront, l0=lFront / 500, maxgrowth=1.1)

    dgrid = DeepGrid(length=lFront*0.1, lIni=fgrid.zjp12[-1] - fgrid.zjm12[-1], z0=lFront)

    plotGrids(fgrid, dgrid)
