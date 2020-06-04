import numpy as np
import matplotlib.pyplot as plt
from material import NonAblativeMaterial

def compareToAnalytical(t, Tnu, inputvars, nmax=201):
    T0 = inputvars.initValue
    q = inputvars.BCfrontValue
    L = inputvars.layers[0].thickness
    mat = inputvars.layers[0].material
    rho = mat.rho(T0)
    cv = mat.cp(T0)
    k = mat.k(T0)
    alpha = k/(rho*cv)
    grid = inputvars.layers[0].grid
    z = grid.zj
    n = np.arange(1, nmax)

    if type(mat) is NonAblativeMaterial and len(mat.data.Tforcp) == 2 and len(mat.data.Tfork) == 2:
        if mat.data.cp[1]/mat.data.cp[0] == mat.data.k[1]/mat.data.k[0] and \
           mat.data.Tforcp[0] == mat.data.Tfork[0] and mat.data.Tforcp[1] == mat.data.Tfork[1]:
            T1 = mat.data.Tforcp[0]
            T2 = mat.data.Tforcp[1]
            k1 = mat.data.k[0]
            k2 = mat.data.k[1]
            Tf = lambda theta : T1 + (T2-T1) * (k1/(k2-k1)) * (-1 + np.sqrt(1 + 2*theta/(T2-T1)*(k2-k1)/k1))
            thetaf = lambda T : (T-T1) + (k2-k1)/(T2-T1) / (2*k1) * (T-T1)**2
    elif type(mat) is NonAblativeMaterial and len(mat.data.Tforcp) == 1 and len(mat.data.Tfork) == 1:
        k1 = mat.data.k[0]
        Tf = lambda theta : theta
        thetaf = lambda T : T
    else:
        print("Unsupported material.")

    th0 = thetaf(T0)

    theta = th0 + q*L/k1 * (alpha*t/L**2 + 1/3 - z/L + 1/2 * (z/L)**2
                            - 2/(np.pi**2) * np.sum(1/n**2 * np.exp(-n**2 * np.pi**2 * alpha*t/(L**2)) *
                                                    np.cos(np.outer(z, n)*np.pi/L), axis=1))

    # Alternative with explicit loop
    #thetaalt = th0 + q*L/k1 * (alpha*t/L**2 + 1/3 - z/L + 1/2 * (z/L)**2)
    #for i in n:
    #    thetaalt += q*L/k1 * (- 2/(np.pi**2) * 1/i**2 * np.exp(-i**2 * np.pi**2 * alpha*t/(L**2) *
    #                                                           np.cos(z*i*np.pi/L)))
    # Talt = Tf(thetaalt)

    T = Tf(theta)

    plt.clf()
    plt.plot(grid.zj, Tnu, label='Numerical')
    plt.plot(z, T, label='Analytical')
    plt.xlabel('z [m]')
    plt.ylabel('T [K]')
    plt.title('Comparison at t = %.3fs' % t)
    plt.legend()
    plt.show()