"""
The output module provides function related to producing output from Hot-STARSHIP,
but also reading that output again and plotting the results. For this, the
SolutionWriter object writes an empty .csv file on initialization and can be called
in the main function for output. The SolutionReader object reads said .csv file
and uses the .plot functions to display temperature, extent of reaction, density,
pyrolysis gas and char mass flux etc.
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime
from matplotlib import lines
from matplotlib.colors import LinearSegmentedColormap
import os
import warnings


def plotT(layers, Tnu, Tmap, t, inputvars):
    """
plots internal temperatures (only to be used while running Hot-STARSHIP,
for post-processing use SolutionReader and its plot capability)
    Parameters
    ----------
    layers : list
        list of layers
    Tnu : np.ndarray
        recession rate and temperature array
    Tmap : dict
        maps layers to indices in Tnu array
    t : float
        time at which to plot
    inputvars : Input
        input variables object

    Returns
    -------

    """

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
    """
plots extent of reaction (only to be used while running Hot-STARSHIP,
for post-processing use SolutionReader and its plot capability)
    Parameters
    ----------
    layers : list
        list of layers
    rhonu : np.ndarray
        nodal densities
    rhomap : dict
        maps layers to indices in rhonu array
    t : float
        time

    Returns
    -------

    """

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
    """ """
    def __init__(self, file, layers, Tmap, inputvars, force_write=False):
        """
checks if file exists, creates file and writes header
        Parameters
        ----------
        file : str
            path to file
        layers : list
            list of layers
        Tmap : dict
            maps layers to indices in Tnu array
        inputvars : Input
            input variables
        force_write : bool
            True if existing files shall be overwritten, False if the user shall be asked first
        """
        # Construct path to write to
        self.cwd = Path.cwd()
        if file is None:
            file = "Hot-STARSHIP_out_" + datetime.today().strftime('%Y-%m-%d-%H:%M:%S') + ".csv"
        self.filepath = Path.joinpath(self.cwd, file)

        # Count number of values to write per time step
        nLayers = len(layers)
        self.nVals = Tmap["lay" + str(nLayers-1)][-1]
        if not layers[0].ablative:
            self.nVals += 1
        if inputvars.BCbackType == "adiabatic":
            self.nVals += 1
        else:
            raise ValueError("Unimplemented back BC %s" % inputvars.BCbackType)

        # Check if file exists
        if not force_write:
            self.checkIfFileExists()

        with open(self.filepath, 'w') as f:
            f.write('t [s]       ;'  # index 0
                    'z [m]       ;'  # index 1
                    'T [K]       ;'  # index 2
                    'rho [kg/m^3];'  # index 3
                    'wv [-]      ;'  # index 4
                    'beta [-]    ;'  # index 5
                    'mc[kg/m^2/s];'  # index 6
                    'mg[kg/m^2/s]'   # index 7
                    '\n')

    def checkIfFileExists(self):
        """ checks if output file already exists, else prompts user to specify a different file"""

        # Check if file exists
        if self.filepath.is_file():
            while True:
                try:
                    inp = str(input("Output file %s already exists.\nDo you wish to overwrite?"
                                    " (Type y/yes or n/no)" % self.filepath))
                except ValueError:
                    print("Invalid input.")
                break

            # If overwrite is not desired, give user the ability to write to different file
            if inp in ("n", "no"):
                inp = str(input("Please specify the file you wish to write to:"))
                self.filepath = Path.joinpath(self.cwd, inp)
                self.checkIfFileExists()
            # If overwrite is desired, go on
            elif inp in ("y", "yes"):
                pass
            # No valid input
            else:
                print("Invalid input.")
                self.checkIfFileExists()

    def write(self, t, layers, Tnu, rhonu, Tmap, rhomap, mgas):
        """
writes to output file
        Parameters
        ----------
        t : float
            time at which to write
        layers : list
            list of layers
        Tnu : np.ndarray
            contains solved temperature solution
        rhonu : np.ndarray
            contains solved density solution
        Tmap : dict
            maps layers to indices in Tnu array
        rhomap : dict
            maps layers to indices in rhonu array
        mgas : np.ndarray
            gas mass flux array

        Returns
        -------

        """

        # Allocate space for variables to be written
        writevars = np.empty((self.nVals, 8))

        # Add t
        writevars[:, 0] = t

        # Add other variables
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
                if layers[iLay].ablative:
                    if mat.data.rhov0 != mat.data.rhoc0:
                        writevars[writelocs, 5] = (mat.data.rhov0 - rhonu[rholocs])/(mat.data.rhov0 - mat.data.rhoc0)
                    else:
                        writevars[writelocs, 5] = 0
                    writevars[writelocs, 6] = 0
                    writevars[0, 6] = rhonu[rholocs][0] * Tnu[Tmap['sdot']]
                    writevars[writelocs, 7] = mgas[::-1].cumsum()[::-1]
                else:
                    writevars[writelocs, 5] = 0
                    writevars[writelocs, 6] = 0
                    writevars[writelocs, 7] = 0
            if key[0:3] == "int":
                iLay = int(key[3:])
                writevars[writelocs, 1] = layers[iLay].grid.zjp12[-1]
                writevars[writelocs, 2] = Tnu[Tlocs]
                writevars[writelocs, 3] = rhonu[rhomap["lay%i" % iLay]][-1]
                writevars[writelocs, 4] = layers[iLay].wv[-1] if layers[iLay].ablative else 1.0
                if layers[iLay].ablative:
                    if mat.data.rhov0 != mat.data.rhoc0:
                        writevars[writelocs, 5] = ((mat.data.rhov0 - rhonu[rhomap["lay%i" % iLay]][-1]) /
                                                   (mat.data.rhov0 - mat.data.rhoc0))
                    else:
                        writevars[writelocs, 5] = 0
                else:
                    writevars[writelocs, 5] = 0
                writevars[writelocs, 6] = 0
                writevars[writelocs, 7] = 0

        lastlay = layers[-1]
        writevars[-1, 1] = lastlay.grid.zjp12[-1]
        writevars[-1, 2] = Tnu[-1]
        writevars[-1, 3] = rhonu[-1]
        writevars[-1, 4] = lastlay.wv[-1] if lastlay.ablative else 1.0
        if layers[iLay].ablative:
            if mat.data.rhov0 != mat.data.rhoc0:
                writevars[-1, 5] = ((mat.data.rhov0 - rhonu[-1]) /
                                    (mat.data.rhov0 - mat.data.rhoc0))
            else:
                writevars[-1, 5] = 0
        else:
            writevars[-1, 5] = 0
        writevars[-1, 6] = 0
        writevars[-1, 7] = 0

        # Save to file
        with open(self.filepath, 'ab') as f:
            np.savetxt(f, writevars, delimiter=';', fmt='%.6E')


def calc_weight(layers, rhonu, rhomap):
    """
calculates weight of TPS
    Parameters
    ----------
    layers : list
        list of layers
    rhonu : np.ndarray
        solved densities
    rhomap :
        maps layers to indices in rhonu array

    Returns
    -------

    """

    weight = 0
    for lay, rhokey in zip(layers, rhomap):
        weight += np.sum((lay.grid.zjp12 - lay.grid.zjm12) * rhonu[rhomap[rhokey]])

    return weight


class SolutionReader:
    def __init__(self, file):
        """
        reads solution csv file and stores the information in arrays

        Parameters
        ----------
        file : str
            file to read
        """
        # Construct path to read from
        cwd = Path.cwd()
        self.filepath = Path.joinpath(cwd, file)
        with open(self.filepath, 'rb') as f:
            loadvars = np.loadtxt(f, skiprows=1, delimiter=';')

        # Get indices of time (where time changes first)
        iTs = np.vstack((0, np.argwhere(np.diff(loadvars[:, 0]) != 0) + 1))

        # Fill arrays with data
        self.t = loadvars[iTs, 0].flatten()
        self.nts = len(self.t)
        self.nVals = int(iTs[1])
        self.z = loadvars[:, 1].reshape(self.nVals, self.nts, order='F')
        self.T = loadvars[:, 2].reshape(self.nVals, self.nts, order='F')
        self.rho = loadvars[:, 3].reshape(self.nVals, self.nts, order='F')
        self.wv = loadvars[:, 4].reshape(self.nVals, self.nts, order='F')
        self.beta = loadvars[:, 5].reshape(self.nVals, self.nts, order='F')
        self.mc = loadvars[:, 6].reshape(self.nVals, self.nts, order='F')
        self.mg = loadvars[:, 7].reshape(self.nVals, self.nts, order='F')
        self.x = self.z - np.tile(self.z[0, :], (self.nVals, 1))

        # Construct dicts for plotting
        self.namedict = {'t': self.t,
                         'z': self.z,
                         'T': self.T,
                         'rho': self.rho,
                         'wv': self.wv,
                         'beta': self.beta,
                         's': self.z,
                         'sdot': self.z,
                         'mc': self.mc,
                         'mg': self.mg,
                         'x': self.x}

        self.labeldict = {'t': r'$t$ [s]',
                          'z': r'$z$ [mm]',
                          'T': r'$T$ [K]',
                          'rho': r'$\rho$ $\left[ \frac{kg}{m^3} \right]$',
                          'wv': r'$w_v$ [-]',
                          'beta': r'$\beta$ [-]',
                          's': '$s$ [mm]',
                          'sdot': r'$\dot{s}$ [mm/s]',
                          'mc': r'$\dot{m}_c$ $\left[ \frac{kg}{m^2 \cdot s} \right]$',
                          'mg': r'$\dot{m}_g$ $\left[ \frac{kg}{m^2 \cdot s} \right]$',
                          'x': r'$x$ [mm]'}

    def plot(self, x_axis, y_axis, t=None, z=None, x=None):
        """

        Parameters
        ----------
        x_axis : str
            variable to plot on x axis
        y_axis : str
            variable to plot on y axis
        t : float, np.ndarray
            time(s) at which to plot (Default value = None)
        z : float, np.ndarray
            stationary location(s) at which to plot (Default value = None)
        x : float, np.ndarray
            moving location(s) at which to plot (Default value = None)

        Returns
        -------

        """

        # For plot of s, use z coordinate of wall
        if y_axis in ('s', 'sdot', 'mc'):
            loc = 'Wall'
            self.loc = self.z

        # Manipulate parameter if needed (convert to numpy array)
        if x_axis in ('z', 'x'):
            location_dependent = True
            if type(t) is float or type(t) is int:
                t = np.array([t])
            elif type(t) is np.ndarray:
                t = t.flatten()
            elif type(t) is list:
                t = np.array(t)
            else:
                raise ValueError("Unknown input type %s" % type(t))
        elif 't' == x_axis:
            location_dependent = False
            if z is not None:
                loc = z
                self.loc = self.z
                leg = self.labeldict['z']
            elif x is not None:
                loc = x
                self.loc = self.x
                leg = self.labeldict['x']

            if type(loc) is float or type(loc) is int:
                loc = np.array([loc])
            elif type(loc) is np.ndarray:
                loc = loc.flatten()
            elif type(loc) is list:
                loc = np.array(loc)
            elif loc == 'Wall':
                loc = np.array([loc])
            else:
                raise ValueError("Unknown input type %s" % type(loc))
        else:
            raise ValueError('%s cannot be on x axis' % x_axis)

        # Get variables
        xvar = self.namedict[x_axis]
        yvar = self.namedict[y_axis]

        # Calculate sdot if selected
        if y_axis == 'sdot':
            yvar = np.gradient(yvar[0, :], xvar)

        # For plots over z
        if location_dependent:
            if any(t < self.t[0]) or any(t > self.t[-1]):
                raise ValueError('Time out of bounds.')
            else:

                # Get indices and times where time lies inbetween
                iTplus = np.argmax(self.t.reshape(self.nts, 1) - t.reshape(1, -1) > 0, axis=0)
                iTminus = iTplus - 1
                tplus = self.t[iTplus]
                tminus = self.t[iTminus]

                # Construct weights for interpolation based on time
                wminus = (tplus - t)/(tplus - tminus)
                wplus = 1 - wminus

                # Fill into global array
                weights = np.zeros((len(t), self.nts))
                weights[np.arange(weights.shape[0]), iTplus] = wplus
                weights[np.arange(weights.shape[0]), iTminus] = wminus

                # Calculate interpolated values
                xvals = np.dot(xvar, weights.transpose())
                yvals = np.dot(yvar, weights.transpose())

        # For plots over t
        else:

            # Check if 'Wall' is part of list
            loclist = loc.tolist()
            if 'Wall' in loclist:
                loc_nowall = loc[loc != 'Wall'].astype(float)
                walls = loc == 'Wall'
                loc_legend = loc
                loc_legend[loc_legend != 'Wall'] = loc_legend[loc_legend != 'Wall'].astype(float)*1e3
            else:
                loc_nowall = loc
                walls = [False] * loc.size
                loc_legend = loc * 1e3


            # Check if valid locations (in material at least at start)
            if any(loc_nowall < self.loc[0, 0]) or any(loc_nowall > self.loc[-1, 0]):
                raise ValueError('Location out of bounds.')
            else:

                # Get indices of location and z coordinate where points lay inbetween
                iZplus = np.argmax(self.loc.reshape(self.nVals, self.nts, 1) - loc_nowall.reshape(1, 1, -1) > 0, axis=0)
                iZminus = iZplus - 1
                locplus = self.loc[iZplus, np.repeat(np.arange(self.nts)[:, np.newaxis], len(loc_nowall), axis=1)]
                locminus = self.loc[iZminus, np.repeat(np.arange(self.nts)[:, np.newaxis], len(loc_nowall), axis=1)]

                # Calculate weights
                wminus = (locplus - loc_nowall) / (locplus - locminus)
                wplus = 1 - wminus

                # Remove points when the material is gone
                with np.errstate(invalid='ignore'):
                    wplus[wminus < 0], wminus[wminus < 0] = (np.nan, np.nan)
                    wplus[wminus > 1], wminus[wminus > 1] = (np.nan, np.nan)
                    wplus[wplus < 0], wminus[wplus < 0] = (np.nan, np.nan)
                    wplus[wplus > 1], wminus[wplus > 1] = (np.nan, np.nan)

                # Fill global weights array and compute interpolated values
                weights = np.zeros((loc.size, self.nVals, self.nts))
                yvals = np.zeros((loc.size, self.nts))
                wall_occurences = 0
                for i, wall in zip(np.arange(loc.size), walls):
                    if wall:
                        weights[i, 0, np.arange(self.nts)] = 1.0
                        wall_occurences += 1
                    else:
                        wo = wall_occurences
                        weights[i, iZplus[:, i-wo], np.arange(self.nts)] = wplus[:, i-wo]
                        weights[i, iZminus[:, i-wo], np.arange(self.nts)] = wminus[:, i-wo]
                    yvals[i, :] = np.sum(yvar*weights[i, :, :], axis=0)

                xvals = np.repeat(self.t[:, np.newaxis], loc.size, axis=1)
                yvals = yvals.transpose()


        # Get plot style
        hs_dir = os.getenv("HOTSTARSHIP_DIR")
        if hs_dir is not None:
            try:
                plt.style.use(str(Path.joinpath(Path(hs_dir), "Templates", "HS_Style.mplstyle")))
            except OSError:
                pass
        else:
            try:
                plt.style.use("HS_Style")
            except OSError:
                warnings.warn("Could not locate HS_Style.mplstyle. Proceeding without.")

        #plt.style.use("Templates/MA_Style")
        #if not vary_linestyle:
        #    plt.plot(self.t, yvals)
        #else:
        #    linestyles = get_linestyles(yvals.shape[1])
        #    for i in range(yvals.shape[1]):
        #        plt.plot(xvals[:, i], yvals[:, i], linestyles[i])
        # colors = ['#332288', '#88CCEE', '#44AA99', '#117733', '#999933', '#DDCC77', '#CC6677', '#882255', '#AA4499']

        # Clear plot
        plt.plot(1, 1)  # dummy plot
        plt.clf()

        #  Generate plot
        for i in range(yvals.shape[1]):
            to_plot = ~np.isnan(yvals[:, i])
            if location_dependent:
                plt.plot(xvals[to_plot, i]*1e3, yvals[to_plot, i])  # , c=colors[i], ls='solid', marker='None')
            else:
                if y_axis in ('sdot', 's'):
                    plt.plot(xvals[to_plot, i], yvals[to_plot, i]*1e3)
                else:
                    plt.plot(xvals[to_plot, i], yvals[to_plot, i])
        if yvals.shape[1] > 1:
            if location_dependent:
                plt.legend(t.astype(str), title=r'$t$ [s]')
            else:
                # TODO: add truncation for ints
                plt.legend(loc_legend.astype(str), title=leg)
        plt.xlabel(self.labeldict[x_axis])
        plt.ylabel(self.labeldict[y_axis])

        plt.show()

        # data_exp = np.transpose(np.vstack((xvals[to_plot, i], yvals[to_plot, i])))
        # np.savetxt('out.csv', data_exp, delimiter=';')

    def plot_gradient(self, t):
        """
        plots the material cross section to give a visual hint of where the material has pyrolyzed

        Parameters
        ----------
        t : float, np.ndarray
            time(s) at which to plot

        Returns
        -------

        """

        if type(t) is float or type(t) is int:
            t = np.array([t])
        elif type(t) is np.ndarray:
            t = t.flatten()
        elif type(t) is list:
            t = np.array(t)
        else:
            raise ValueError("Unknown input type %s" % type(t))

        num_t = len(t)

        # Get variables
        xvar = self.namedict['z']
        yvar = self.namedict['beta']

        if any(t < self.t[0]) or any(t > self.t[-1]):
            raise ValueError('Time out of bounds.')
        else:

            # Get indices and times where time lies inbetween
            iTplus = np.argmax(self.t.reshape(self.nts, 1) - t.reshape(1, -1) > 0, axis=0)
            iTminus = iTplus - 1
            tplus = self.t[iTplus]
            tminus = self.t[iTminus]

            # Construct weights for interpolation based on time
            wminus = (tplus - t) / (tplus - tminus)
            wplus = 1 - wminus

            # Fill into global array
            weights = np.zeros((len(t), self.nts))
            weights[np.arange(weights.shape[0]), iTplus] = wplus
            weights[np.arange(weights.shape[0]), iTminus] = wminus

            # Calculate interpolated values
            xvals = np.dot(xvar, weights.transpose())
            yvals = np.dot(yvar, weights.transpose())

        height = np.max(xvals)
        width = height/2

        # Colormap
        cvals = [0, 1]
        colors = ["#cac4a9", "black"]
        norm = plt.Normalize(min(cvals), max(cvals))
        tuples = list(zip(map(norm, cvals), colors))
        cmap = LinearSegmentedColormap.from_list("", tuples)

        # Subplots
        fig = plt.figure(1)
        for it, cur_t in enumerate(t):
            X, Y = np.meshgrid(np.array([0, width]), xvals[:, it])
            Z = np.tile(yvals[:, it], (2, 1)).transpose()

            ax = fig.add_subplot(1, num_t, it+1)
            ax.contourf(X, Y, Z, levels=998, cmap=cmap, vmin=0, vmax=1)
            ax.set_aspect('equal', adjustable='box')
            plt.xlim([0, width])
            plt.ylim([0, height])
            ax.invert_yaxis()
            plt.xticks([])
            plt.yticks([])
            plt.box(False)
            if int(cur_t) == cur_t:
                ax.set_xlabel('%s s' % int(cur_t))
            else:
                ax.set_xlabel('%s s' % cur_t)

        plt.show()


    def calculate_mass(self, t=0.0):
        """
        calculates TPS mass at specified time

        Parameters
        ----------
        t : float
            time (Default value = 0.0)

        Returns
        -------
        float
            TPS mass
        """

        if type(t) in (int, float):
            t = np.array([t])

        # Get indices and times where time lies inbetween
        iTplus = np.argmax(self.t.reshape(self.nts, 1) - t.reshape(1, -1) > 0, axis=0)
        iTminus = iTplus - 1
        tplus = self.t[iTplus]
        tminus = self.t[iTminus]

        # Construct weights for interpolation based on time
        wminus = (tplus - t) / (tplus - tminus)
        wplus = 1 - wminus

        # Fill into global array
        weights = np.zeros((len(t), self.nts))
        weights[np.arange(weights.shape[0]), iTplus] = wplus
        weights[np.arange(weights.shape[0]), iTminus] = wminus

        # Calculate interpolated values
        rho_at_t = np.dot(self.rho, weights.transpose())
        z_at_t = np.dot(self.z, weights.transpose())

        zjp12 = z_at_t[+1:, :]
        zjm12 = z_at_t[:-1, :]

        mass = np.sum(rho_at_t[:-1] * (zjp12 - zjm12), axis=0).flatten()

        return mass

    def get_max_back_T(self):
        """
        returns maximum back face temperature over time

        Returns
        -------
        float
            maximum back face temperature
        """

        return np.max(self.T[-1, :])

    def get_max_T(self):
        """
        returns maximum global temperature over time

        Returns
        -------
        float
            maximum global temperature
        """

        return np.max(self.T)

    def get_remaining_thickness(self):
        """
        calculates remaining thickness of TPS at end of calculation

        Returns
        -------
        float
            returns remaining thickness of TPS
        """

        return self.z[-1, -1] - self.z[0, -1]


def get_linestyles(n):
    """
depreceated
    Parameters
    ----------
    n :
        

    Returns
    -------

    """
    nAllowable = 4
    possible_styles = list(lines.lineStyles.keys())[:nAllowable]
    return [possible_styles[i % nAllowable] for i in range(n)]
