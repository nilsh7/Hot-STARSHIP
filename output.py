import matplotlib.pyplot as plt
import numpy as np


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
