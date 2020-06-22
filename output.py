import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime
from matplotlib import lines


def plotT(layers, Tnu, Tmap, t, inputvars):

    plt.clf()
    ztot = np.array([])
    Ttot = np.array([])
    for key in Tmap:
        if key[0:3] == "lay":
            iLay = int(key[3:])
            ztot = np.append(ztot, layers[iLay].grid.zj)
            Ttot = np.append(Ttot, Tnu[Tmap[key]])
            if "int" + key[3:] not in Tmap:
                # Last layer
                if inputvars.BCbackType == "adiabatic":
                    ztot = np.append(ztot, layers[iLay].grid.zjp12[-1])
                    Ttot = np.append(Ttot, Ttot[-1])
                else:
                    raise UserWarning("Unknown back BC %s" % inputvars.BCbackType)
        if key[0:3] == "int":
            iInt = int(key[3:])
            ztot = np.append(ztot, layers[iInt].grid.zjp12[-1])
            Ttot = np.append(Ttot, Tnu[Tmap[key]])
    plt.plot(ztot, Ttot, '-o', label='Numerical solution')
    plt.xlabel('z [m]')
    plt.ylabel('T [K]')
    plt.title('Solution at t = %.3fs' % t)
    plt.show()


def plotBeta(layers, rhonu, rhomap, t):

    lay = layers[0]
    if not lay.ablative:
        raise UserWarning("Cannot plot beta for non-ablative material.")
        return
    mat = lay.material
    gr = lay.grid
    rhoj = rhonu[rhomap["lay0"]]
    beta = (mat.data.rhov0 - rhoj)/(mat.data.rhov0 - mat.data.rhoc0)
    beta_int = beta[-2] + (beta[-1] - beta[-2])/(gr.zj[-1] - gr.zj[-2]) * (gr.zjp12[-1] - gr.zj[-2])
    beta_tot = np.hstack((beta, beta_int))
    z_tot = np.hstack((gr.zj, gr.zjp12[-1]))

    # Construct plot
    plt.clf()
    plt.plot(z_tot, beta_tot, '-o', label='Numerical solution')
    plt.xlabel('z [m]')
    plt.ylabel('beta [-]')
    plt.title('Solution at t = %.3fs' % t)
    plt.show()


class SolutionWriter:
    def __init__(self, file, layers, Tmap, inputvars):

        # Construct path to write to
        cwd = Path.cwd()
        if file is None:
            file = "Hot-STARSHIP_out_" + datetime.today().strftime('%Y-%m-%d-%H:%M:%S') + ".csv"
        self.filepath = Path.joinpath(cwd, file)

        # Count number of values to write per time step
        nLayers = len(layers)
        self.nVals = Tmap["lay" + str(nLayers-1)][-1]
        if not layers[0].ablative:
            self.nVals += 1
        if inputvars.BCbackType == "adiabatic":
            self.nVals += 1
        else:
            raise ValueError("Unimplemented back BC %s" % inputvars.BCbackType)

        # Clear file if existent
        with open(self.filepath, 'w') as f:
            f.write('t [s]       ;'
                    'z [m]       ;'
                    'T [K]       ;'
                    'rho [kg/m^3];'
                    'wv [-]      ;'
                    'beta [-]    '
                    '\n')

    def write(self, t, layers, Tnu, rhonu, Tmap, rhomap):

        # Allocate space for variables to be written
        writevars = np.empty((self.nVals, 6))

        # Add t
        writevars[:, 0] = t

        for key in Tmap:
            writelocs = Tmap[key] if not layers[0].ablative else Tmap[key] - 1
            Tlocs = Tmap[key]
            if key[0:3] == "lay":
                iLay = int(key[3:])
                mat = layers[iLay].material
                rholocs = rhomap[key]
                writevars[writelocs, 1] = layers[iLay].grid.zj
                writevars[writelocs, 2] = Tnu[Tlocs]
                writevars[writelocs, 3] = rhonu[rholocs]
                writevars[writelocs, 4] = layers[iLay].wv if layers[iLay].ablative else 1.0
                writevars[writelocs, 5] = (mat.data.rhov0 - rhonu[rholocs])/(mat.data.rhov0 - mat.data.rhoc0)\
                    if layers[iLay].ablative else 0.0
            if key[0:3] == "int":
                iLay = int(key[3:])
                writevars[writelocs, 1] = layers[iLay].grid.zjp12[-1]
                writevars[writelocs, 2] = Tnu[Tlocs]
                writevars[writelocs, 3] = rhonu[rhomap["lay%i" % iLay]][-1]
                writevars[writelocs, 4] = layers[iLay].wv[-1] if layers[iLay].ablative else 1.0
                writevars[writelocs, 5] = ((mat.data.rhov0 - rhonu[rhomap["lay%i" % iLay]][-1]) /
                                           (mat.data.rhov0 - mat.data.rhoc0)) if layers[iLay].ablative else 0.0

        lastlay = layers[-1]
        writevars[-1, 1] = lastlay.grid.zjp12[-1]
        writevars[-1, 2] = Tnu[-1]
        writevars[-1, 3] = rhonu[-1]
        writevars[-1, 4] = lastlay.wv[-1] if lastlay.ablative else 1.0
        writevars[-1, 5] = ((mat.data.rhov0 - rhonu[-1]) /
                            (mat.data.rhov0 - mat.data.rhoc0)) if lastlay.ablative else 0.0

        # Save to file
        with open(self.filepath, 'ab') as f:
            np.savetxt(f, writevars, delimiter=';', fmt='%.6E')


class SolutionReader:
    def __init__(self, file):

        # Construct path to read from
        cwd = Path.cwd()
        self.filepath = Path.joinpath(cwd, file)
        with open(self.filepath, 'rb') as f:
            loadvars = np.loadtxt(f, skiprows=1, delimiter=';')

        iTs = np.vstack((0, np.argwhere(np.diff(loadvars[:, 0]) != 0) + 1))

        self.t = loadvars[iTs, 0].flatten()
        self.nts = len(self.t)
        self.nVals = int(iTs[1])
        self.z = loadvars[:, 1].reshape(self.nVals, self.nts, order='F')
        self.T = loadvars[:, 2].reshape(self.nVals, self.nts, order='F')
        self.rho = loadvars[:, 3].reshape(self.nVals, self.nts, order='F')
        self.wv = loadvars[:, 4].reshape(self.nVals, self.nts, order='F')
        self.beta = loadvars[:, 5].reshape(self.nVals, self.nts, order='F')

        self.namedict = {'t': self.t,
                         'z': self.z,
                         'T': self.T,
                         'rho': self.rho,
                         'wv': self.wv,
                         'beta': self.beta}

        self.labeldict = {'t': 't [s]',
                          'z': 'z [m]',
                          'T': 'T [K]',
                          'rho': 'rho [kg/m^3]',
                          'wv': 'wv [-]',
                          'beta': 'beta [-]'}

    def plot(self, x, y, t=0.0, z=0.0, varLS=False):

        if 'z' == x:
            location_dependent = True
            if type(t) is float:
                t = np.array([t])
            elif type(t) is np.ndarray:
                t = t.flatten()
            elif type(t) is list:
                t = np.array(t)
            else:
                raise ValueError("Unknown input type %s" % type(t))
        elif 't' == x:
            location_dependent = False
        else:
            raise ValueError('%s cannot be on x axis' % x)

        xvar = self.namedict[x]
        yvar = self.namedict[y]

        if location_dependent:
            if any(t < self.t[0]) or any(t > self.t[-1]):
                raise ValueError('Time out of bounds.')
            else:
                iTplus = np.argmax(self.t.reshape(self.nts, 1) - t.reshape(1, -1) > 0, axis=0)
                iTminus = iTplus - 1
                tplus = self.t[iTplus]
                tminus = self.t[iTminus]
                wminus = (tplus - t)/(tplus - tminus)
                wplus = 1 - wminus

                weights = np.zeros((len(t), self.nts))
                weights[np.arange(weights.shape[0]), iTplus] = wplus
                weights[np.arange(weights.shape[0]), iTminus] = wminus

                xvals = np.dot(xvar, weights.transpose())
                yvals = np.dot(yvar, weights.transpose())

                if not varLS:
                    plt.plot(xvals, yvals)
                else:
                    linestyles = list(lines.lineStyles.keys())[:len(t)]
                    for i in range(yvals.shape[1]):
                        plt.plot(xvals[:, i], yvals[:, i], linestyles[i])
                if len(t) > 1:
                    plt.legend(t.astype(str), title='t [s]')
                plt.xlabel(self.labeldict[x])
                plt.ylabel(self.labeldict[y])
                plt.grid()
                plt.show()
