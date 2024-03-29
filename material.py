"""
The material module provides function related to reading and providing material data.
It reads material data in form of .csv files in a set directory structure.
For ablative materials, it runs Mutation++ to generate B' tables and pyrolysis gas
enthalpy data. The Material objects hold piecewise linear splines of conductivity,
heat capacity, emissivity and internal energy as a function of temperature and
virgin weight fraction.
It can be invoked with the main function at the bottom to produce material properties
.matp files that are especially useful for ablative materials by saving the time to
run Mutation++ on each run or use these files on Windows machines.
"""
from pathlib import Path
import pandas as pd
import warnings
import numpy as np
import argparse
import re
from itertools import compress
import dill  # for pickling lambdas
import os
from io import StringIO
import matplotlib.pyplot as plt
from scipy import interpolate as ip
from scipy.integrate import cumtrapz
import input
import math

# previously used packages: csv, os, glob, plt, json, jsonpickle, pickle, sympy

nonsiunits = [unit.upper() for unit in ("ft", "°R", "BTU", "lb", "°F", "°C", "cal", "hr")]
whitelist = [word.upper() for word in ("Scaled",)]

converters = {"-": 0, "#WERT!": 0}

convertf = lambda st: "0,0" if st in ("-", "#WERT!") else st

try:
    hotstarship_dir = os.environ["HOTSTARSHIP_DIR"]
except KeyError:
    raise IOError("Environment variable \'HOTSTARSHIP_DIR\' could not be found.\n"
                  "Please export the variable.")


class Material:
    """ """
    def __init__(self):
        self.data = Data()

    def readFile(self):
        """ """
        pass  # to be initiated in Ablative or NonAblativeMaterial

    def calculateVariables(self):
        """creates functions for thermophysical properties"""
        pass  # to be initiated in Ablative or NonAblativeMaterial

    def readData(self, args):
        """loops over directories to be read and initiates reading procedure

        Parameters
        ----------
        args : dict
            dictionary of input arguments

        Returns
        -------

        """
        # Construct path to current material and check existent
        if not Path.is_dir(args["input_dir"]):
            raise IOError("Could not locate %s" % args["input_dir"])

        # Loop over all dirs
        for iDir, d in enumerate(self.necessarydirs):

            globdir = Path.joinpath(args["input_dir"], d)

            # Check if necessary directory exists
            if not Path.is_dir(globdir):
                raise IOError("Directory %s could not be located in %s" % (d, args["input_dir"]))
            else:

                # Check whether there is more than one csv file
                csv_files = list(globdir.glob("*.csv"))
                if d.name != "Combined" and len(csv_files) > 1:
                    raise IOError("Located more than one csv file in %s" % d)

                else:
                    self.readFile(iDir, args["input_dir"], csv_files)


class State:
    """for allocating variables to save data to """
    def __init__(self):
        self.data = Data()


class Data:
    """provides a place to save data """
    def __init__(self):
        pass


class NonAblativeMaterial(Material):
    """ """
    def __init__(self, args):
        """
        saves non-ablative material properties

        Parameters
        ----------
        args : dict
            dictionary of input variables
        """

        print("Constructing non-ablative material \"%s\"..." % args["input_dir"].name)

        Material.__init__(self)

        # List of necessary directories
        self.necessarydirs = [Path(d) for d in ["cp", "eps", "k", "rho"]]

        self.data.Tref = 298.15

        self.readData(args)

        self.calculateVariables()

    def readFile(self, iDir, globdir, csv_files):
        """reads csv file and stores data

        Parameters
        ----------
        iDir : int
            directory number
        globdir : str
            global path to directory
        csv_files : list
            csv files in directory

        Returns
        -------

        """
        # Determine csv file
        csv_file = Path.joinpath(globdir, csv_files[0])

        # Open file
        with open(csv_file) as f:
            data = pd.read_csv(f, sep=';', decimal='.')

        data = dropnafromboth(data)

        checkForNonSI(data, csv_file)

        # Fill arrays with values
        if iDir == 0:
            self.data.Tforcp = data.values[:, 0]
            self.data.cp = data.values[:, 1]
        elif iDir == 1:
            self.data.Tforeps = data.values[:, 0]
            self.data.eps = data.values[:, 1]
        elif iDir == 2:
            self.data.Tfork = data.values[:, 0]
            self.data.k = data.values[:, 1]
        elif iDir == 3:
            self.data.Tforrho = data.values[:, 0]
            self.data.rho = data.values[:, 1]

    def calculateVariables(self):
        """constructs splines for each material property (cp, k, eps, ...) """

        # cp
        self.data.cpLin = constructLinearSpline(self.data.Tforcp, self.data.cp)

        # k
        self.data.kLin = constructLinearSpline(self.data.Tfork, self.data.k)
        self.data.dkdTLin = self.data.kLin.derivative(1)

        # eps
        self.data.epsLin = constructLinearSpline(self.data.Tforeps, self.data.eps)
        self.data.depsdTLin = self.data.epsLin.derivative(1)

        # rho
        self.data.rhoLin = constructLinearSpline(self.data.Tforrho, self.data.rho)

        # e
        self.data.e = self.data.cpLin.antiderivative(1)

    def cp(self, T, *ignoreargs):
        """
        heat capacity

        Parameters
        ----------
        T : np.ndarray
            temperature array
            
        *ignoreargs :
            

        Returns
        -------

        """
        return self.data.cpLin(T)

    def k(self, T, *ignoreargs):
        """
        conductivity

        Parameters
        ----------
        T : np.ndarray
            temperature array
            
        *ignoreargs :
            

        Returns
        -------

        """
        return self.data.kLin(T)

    def eps(self, T, *ignoreargs):
        """
        emissivity

        Parameters
        ----------
        T : np.ndarray
            temperature array
            
        *ignoreargs :
            

        Returns
        -------

        """
        return self.data.epsLin(T)

    def depsdT(self, T, *ignoreargs):
        """
        emissivity gradient

        Parameters
        ----------
        T : np.ndarray
            temperature array
            
        *ignoreargs :
            

        Returns
        -------

        """
        return self.data.depsdTLin(T)

    def rho(self, T, *ignoreargs):
        """
        density

        Parameters
        ----------
        T : np.ndarray
            temperature array
            
        *ignoreargs :
            

        Returns
        -------

        """
        return self.data.rhoLin(T)

    def dkdT(self, T, *ignoreargs):
        """
        conductivity gradient

        Parameters
        ----------
        T : np.ndarray
            temperature array
            
        *ignoreargs :
            

        Returns
        -------

        """
        return self.data.dkdTLin(T)

    def e(self, T, *ignoreargs):
        """
        internal energy

        Parameters
        ----------
        T : np.ndarray
            temperature array
            
        *ignoreargs :
            

        Returns
        -------

        """
        return self.data.e(T)


class CorrugatedMaterial(Material):
    """save material properties for corrugated sandwich cores;
    For the used formulas, see eqs. 1-3 in:
    Gogu, C., Bapanapalli, S. K., Haftka, R. T., & Sankar,B. V. (2009).
    Comparison of materials for an integrated thermal protection system for spacecraft reentry.
    Journal of Spacecraft and Rockets, 46(3), 501–513. https://doi.org/10.2514/1.35669

    """
    def __init__(self, args):
        corrugated_vals = args["corrugated_vals"]

        self.mat_core = input.readMaterial(matname=corrugated_vals["mat_core"], ablative=False, corrugated=False)
        self.mat_web = input.readMaterial(matname=corrugated_vals["mat_web"], ablative=False, corrugated=False)
        self.dc = corrugated_vals["dc"]
        self.dw = corrugated_vals["dw"]
        self.p = corrugated_vals["p"]
        self.theta = corrugated_vals["theta"]

    # For the following formulas, see eqs. 1-3 in: Gogu, C., Bapanapalli, S. K., Haftka, R. T., & Sankar,
    # B. V. (2009). Comparison of materials for an integrated thermal protection system for spacecraft reentry.
    # Journal of Spacecraft and Rockets, 46(3), 501–513. https://doi.org/10.2514/1.35669
    def rho(self, T, *ignoreargs):
        """
        volume weighted density

        Parameters
        ----------
        T : np.ndarray
            temperature array
            
        *ignoreargs :
            

        Returns
        -------

        """
        mw = self.mat_web
        mc = self.mat_core
        return (mw.rho(T) * self.dw + mc.rho(T) * (self.p * math.sin(self.theta) - self.dw)) / (
                self.p * math.sin(self.theta))

    def cp(self, T, *ignoreargs):
        """
        mass weighted heat capacity

        Parameters
        ----------
        T : np.ndarray
            temperature array
            
        *ignoreargs :
            

        Returns
        -------

        """
        mw = self.mat_web
        mc = self.mat_core
        return (mw.rho(T) * mw.cp(T) * self.dw + mc.rho(T) * mc.cp(T) * (self.p * math.sin(self.theta) - self.dw)) / \
               (mw.rho(T) * self.dw + mc.rho(T) * (self.p * math.sin(self.theta) - self.dw))

    def k(self, T, *ignoreargs):
        """
        area weighted conductivity

        Parameters
        ----------
        T : np.ndarray
            temperature array
            
        *ignoreargs :
            

        Returns
        -------

        """
        mw = self.mat_web
        mc = self.mat_core
        return (mw.k(T) * self.dw + mc.k(T) * (self.p * math.sin(self.theta) - self.dw)) / (
                self.p * math.sin(self.theta))

    def dkdT(self, T, *ignoreargs):
        """
        area weighted conductivity gradient

        Parameters
        ----------
        T : np.ndarray
            temperature array
            
        *ignoreargs :
            

        Returns
        -------

        """
        mw = self.mat_web
        mc = self.mat_core
        return (mw.dkdT(T) * self.dw + mc.dkdT(T) * (self.p * math.sin(self.theta) - self.dw)) / (
                self.p * math.sin(self.theta))

    def e(self, T, *ignoreargs):
        """
        mass weighted internal energy

        Parameters
        ----------
        T : np.ndarray
            temperature array
            
        *ignoreargs :
            

        Returns
        -------

        """
        mw = self.mat_web
        mc = self.mat_core
        return (mw.rho(T) * mw.e(T) * self.dw + mc.rho(T) * mc.e(T) * (self.p * math.sin(self.theta) - self.dw)) / \
               (mw.rho(T) * self.dw + mc.rho(T) * (self.p * math.sin(self.theta) - self.dw))

    def eps(self, T, *ignoreargs):
        """
        core emissivity

        Parameters
        ----------
        T : np.ndarray
            temperature array
            
        *ignoreargs :
            

        Returns
        -------

        """
        return self.mat_core.eps(T)

    def depsdT(self, T, *ignoreargs):
        """
        core emissivity gradient

        Parameters
        ----------
        T : np.ndarray
            temperature array
            
        *ignoreargs :
            

        Returns
        -------

        """
        return self.mat_core.depsdT(T)


class AblativeMaterial(Material):
    """stores ablative material properties """
    def __init__(self, args):

        print("Constructing ablative material \"%s\"..." % args["input_dir"].name)

        self.storeVariables(args)

        Material.__init__(self)
        self.virgin = State()
        self.char = State()
        self.gas = State()

        self.necessarydirs = [Path(d) for d in ["Virgin/cp", "Virgin/eps", "Virgin/k", "Virgin/comp",
                                                "Char/cp", "Char/eps", "Char/k",
                                                "Combined"]]

        self.pyroelems = '{gases with C, H, O, e-}'
        self.surfelems = '{gases with C, H, N, O, e-, Ar} C(gr)'
        self.atmoelems = '{gases with C, H, N, O, e-, Ar}'

        self.readData(args)

        self.calculateVariables()

        self.calculateAblativeProperties(args)

        self.calculateEnergies()

    def readFile(self, iDir, globdir, csv_files):
        """reads csv file and stores data

        Parameters
        ----------
        iDir : int
            directory number
        globdir : str
            global path to directory
        csv_files : list
            csv files in directory

        Returns
        -------

        """
        # Open files and read data
        if iDir != 7:
            csv_file = Path.joinpath(globdir, csv_files[0])

            head, index = (None, 0) if iDir == 3 else (0, None)

            # Open file
            with open(csv_file) as f:
                data = pd.read_csv(f, sep=';', decimal='.', header=head, index_col=index)

            data = dropnafromboth(data)

            checkForNonSI(data, csv_file)

        else:
            # Read heats of formation
            hof_file = Path.joinpath(globdir, "Combined", "Heats_of_Formation.csv")

            # Open file
            with open(hof_file) as f:
                hof_data = pd.read_csv(f, sep=';', decimal='.', header=None, index_col=0,
                                       converters={k: convertf for k in range(100)})

            # Replace commas with dots and convert to float
            hof_data = hof_data.apply(lambda x: x.str.replace(',', '.'))
            hof_data = hof_data.apply(lambda x: x.str.replace(r'^\s*$', "NaN", regex=True))
            hof_data = hof_data.astype(np.float64)
            hof_data = dropnafromboth(hof_data)

            checkForNonSI(hof_data, hof_file)

            # Read Decomposition Kinetics
            dec_file = Path.joinpath(globdir, "Combined", "Decomposition_Kinetics.csv")

            # Open file
            with open(dec_file) as f:
                dec_data = pd.read_csv(f, sep=';', decimal=',',
                                       converters={k: convertf for k in range(100)},
                                       # dtype=np.float64,
                                       index_col=0)

            # Replace commas with dots and convert to float
            dec_data = dec_data.apply(lambda x: x.str.replace(',', '.'))
            dec_data = dec_data.apply(lambda x: x.str.replace(r'^\s*$', "NaN", regex=True))
            dec_data = dec_data.astype(np.float64)
            dec_data = dropnafromboth(dec_data)

            checkForNonSI(dec_data, dec_file)

        # Fill arrays with values
        if iDir == 0:
            self.virgin.data.Tforcp = data.values[:, 0]
            self.virgin.data.cp = data.values[:, 1]
        elif iDir == 1:
            self.virgin.data.Tforeps = data.values[:, 0]
            self.virgin.data.eps = data.values[:, 1]
        elif iDir == 2:
            self.virgin.data.Tfork = data.values[:, 0]
            self.virgin.data.k = data.values[:, 1]
        elif iDir == 3:
            # Composition
            self.virgin.data.comp = pd.DataFrame(data=data.values[0:3], index=['C', 'H', 'O'], columns=['initialComp'])
            # self.virgin.data.comp = {"C": data.values[0],
            #                         "H": data.values[1],
            #                         "O": data.values[2]}
            self.virgin.data.gyield = data.values[3]
        elif iDir == 4:
            self.char.data.Tforcp = data.values[:, 0]
            self.char.data.cp = data.values[:, 1]
        elif iDir == 5:
            self.char.data.Tforeps = data.values[:, 0]
            self.char.data.eps = data.values[:, 1]
        elif iDir == 6:
            self.char.data.Tfork = data.values[:, 0]
            self.char.data.k = data.values[:, 1]
        elif iDir == 7:

            # Heats of formation
            self.data.Tref = hof_data.values[0, 0]
            self.virgin.data.hf = hof_data.values[1, 0]
            self.char.data.hf = hof_data.values[2, 0]
            self.gas.data.hf = hof_data.values[3, 0]

            # Decomposition kinetics
            self.data.nDecomp = dec_data.columns.values.size
            self.data.virginRhoFrac0 = dec_data.values[0, :]
            self.data.charRhoFrac0 = dec_data.values[1, :]
            self.data.kr = dec_data.values[2, :]
            self.data.nr = dec_data.values[3, :]
            self.data.Ei = dec_data.values[4, :]
            self.data.Tmin = dec_data.values[5, :]
            self.data.frac = dec_data.values[6, :]
            self.data.rhov0 = np.sum(self.data.virginRhoFrac0 * self.data.frac)
            self.data.rhoc0 = np.sum(self.data.charRhoFrac0 * self.data.frac)

    def calculateVariables(self):
        """creates splines for each material property (cp, k, eps, ...), virgin and char """
        ### Virgin ###
        # cp
        self.virgin.cp = constructLinearSpline(self.virgin.data.Tforcp, self.virgin.data.cp)

        # k
        self.virgin.k = constructLinearSpline(self.virgin.data.Tfork, self.virgin.data.k)
        self.virgin.dkdT = self.virgin.k.derivative(1)

        # eps
        self.virgin.eps = constructLinearSpline(self.virgin.data.Tforeps, self.virgin.data.eps)
        self.virgin.depsdT = self.virgin.eps.derivative(1)

        # e
        # to be calculated later

        ### Char ###
        # cp
        self.char.cp = constructLinearSpline(self.char.data.Tforcp, self.char.data.cp)

        # k
        self.char.k = constructLinearSpline(self.char.data.Tfork, self.char.data.k)
        self.char.dkdT = self.char.k.derivative(1)

        # eps
        self.char.eps = constructLinearSpline(self.char.data.Tforeps, self.char.data.eps)
        self.char.depsdT = self.char.eps.derivative(1)

        # e
        # to be calculated later

        ### Gas ###
        # Gas enthalpy to be determined using mppequil (see calculatePyroGasComposition)

        ### Combined ###
        # cp
        self.cp = lambda T, wv: wv * self.virgin.cp(T) + (1 - wv) * self.char.cp(T)

        # k
        self.k = lambda T, wv: wv * self.virgin.k(T) + (1 - wv) * self.char.k(T)
        self.dkdT = lambda T, wv: wv * self.virgin.dkdT(T) + (1 - wv) * self.char.dkdT(T)

        # eps
        self.eps = lambda T, wv: wv * self.virgin.eps(T) + (1 - wv) * self.char.eps(T)
        self.depsdT = lambda T, wv: wv * self.virgin.depsdT(T) + (1 - wv) * self.char.depsdT(T)

        # e
        # to be calculated later

    def calculateAblativeProperties(self, args):
        """calculate pyrolysis gas composition and bprime table

        Parameters
        ----------
        args : dict
            dictionary of arguments

        Returns
        -------

        """

        if args["hgas"] is None:
            self.calculatePyroGasComposition(args)
        else:
            self.readPyroGasComposition(args)

        if args["bprime"] is None:
            self.calculateBPrimes(args)
        else:
            self.readBPrimes(args)

        self.calculateHatmo(args)

    def calculatePyroGasComposition(self, args):
        """calculates pyrolysis gas composition using mppequil

        Parameters
        ----------
        args : dict
            dictionary of arguments

        Returns
        -------

        """

        print("Calculating pyrolysis gas composition using mppequil...")

        a = self.virgin.data.comp  # for easier understanding of code below

        # Calculate composition of pyrolysis gas based on char yield
        a['virginmolefrac'] = a['initialComp'] / sum(a['initialComp'])
        a['molarmass'] = [12.011, 1.008, 15.999]
        a['virginmass'] = a['virginmolefrac'] * a['molarmass']
        a['pyromass'] = a['virginmass']
        a.at['C', 'pyromass'] = a.at['C', 'virginmass'] - self.virgin.data.gyield * sum(a['virginmass'])
        a['pyroweightfrac'] = a['pyromass'] / sum(a['pyromass'])
        a['pyromoles'] = a['pyromass'] / a['molarmass']
        a['pyromolefrac'] = a['pyromoles'] / sum(a['pyromoles'])

        # Check if negative number
        if any(a["pyromolefrac"] < 0):
            raise ValueError("Negative mole fraction in pyrolysis gas composition!\n"
                             "Lower the expected yield of pyrolysis gas.")

        # Construct Mutation++ mixture file
        xmldata = Path(hotstarship_dir, "Templates/mutationpp_mixtures_pyrogas.xml").read_text()
        gasname = args["input_dir"].name + "_Gas"
        xmldata = xmldata.replace("!name", gasname)  # Replace name input dir name
        xmldata = xmldata.replace("!elems", self.pyroelems)
        moletxt = ""
        for index, row in a.iterrows():
            moletxt += str(row.name) + str(":") + str(row["pyromolefrac"]) + ", "
        xmldata = xmldata.replace("!gascomp", moletxt[:-2])  # Replace mole fractions

        xmlfilename = Path(args["input_dir"], "mutationpp_gas_" + args["input_dir"].name + ".xml")
        with open(xmlfilename, 'w') as xmlfile:
            xmlfile.write(xmldata)

        # Construct mppequil command
        moletxt = ""
        for index, row in a.iterrows():
            moletxt += str(row.name) + str(":") + str(row["pyromolefrac"]) + ","
        mppcmd = "mppequil -T " + args["temprange"] + " -P " + str(args["p"]) + " -m 0,10 " + str(xmlfilename)

        expDYLD = 'export DYLD_LIBRARY_PATH=$MPP_DIRECTORY/install/lib:$DYLD_LIBRARY_PATH\n'
        h = StringIO(os.popen(expDYLD + mppcmd).read())  # Execute mppequil command

        # Read table
        htab = pd.read_table(h, header=0, delim_whitespace=True)

        # Calculate enthalpy
        self.gas.h = constructLinearSpline(x=htab.values[:, 0], y=htab.values[:, 1])
        self.gas.cp = self.gas.h.derivative()

    def readPyroGasComposition(self, args):
        """
        reads pyrolysis gas enthalpy from csv file (first column T, second column hgas)

        Parameters
        ----------
        args : dict
            dictionary of input arguments
            

        Returns
        -------

        """

        csvfile = args["hgas"]

        with open(csvfile) as f:
            data = pd.read_csv(f, sep=';', decimal='.', header=0, index_col=None).to_numpy()

        Ts = data[:, 0]
        hs = data[:, 1]

        # Calculate enthalpy
        self.gas.h = constructLinearSpline(x=Ts, y=hs)
        self.gas.cp = self.gas.h.derivative()

    def calculateBPrimes(self, args):
        """calculates bprime tables using bprime executable provided by mutation++

        Parameters
        ----------
        args : dict
            dictionary of arguments

        Returns
        -------

        """

        print("Calculating wall gas composition using bprime (from Mutation++ library)...")

        # Construct Mutation++ mixture file
        xmldata = Path(hotstarship_dir, "Templates/mutationpp_mixtures_surface.xml").read_text()
        gasname = args["input_dir"].name + "_Gas"
        xmldata = xmldata.replace("!name", gasname)  # Replace name input dir name
        xmldata = xmldata.replace("!elems", self.surfelems)
        moletxt = ""
        for index, row in self.virgin.data.comp.iterrows():
            moletxt += str(row.name) + str(":") + str(row["pyromolefrac"]) + ", "
        xmldata = xmldata.replace("!gascomp", moletxt[:-2])  # Replace mole fractions

        xmlfilename = Path(args["input_dir"], "mutationpp_surf_" + args["input_dir"].name + ".xml")
        with open(xmlfilename, 'w') as xmlfile:
            xmlfile.write(xmldata)

        # Determine b range
        b_low, b_high = (float(args["bg"].split(":")[0]), float(args["bg"].split(":")[2]))
        explow, exphigh = (np.log10(b_low), np.log10(b_high))
        expstep = float(args["bg"].split(":")[1])
        nums = int(round((exphigh - explow) / expstep) + 1)  # Calculate number of values
        b_vals = 10 ** np.linspace(start=explow, stop=exphigh, num=nums)

        # Add fine region where high precision is necessary
        b_fine = np.array([0.3, 0.35, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.48, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        b_onlygas = np.array([1.0e4])
        b_vals = b_vals[np.logical_not(np.logical_and(min(b_fine) - 1.0e-6 < b_vals,
                                                      b_vals < max(b_fine) + 1.0e-6))]  # Delete b values in this region
        b_vals = np.concatenate((b_vals, b_fine, b_onlygas))
        b_vals.sort()

        # Construct empty numpy arrays for bc and hw
        T_low, T_step, T_high = [float(T) for T in args["temprange"].split(":")]
        numTs = int(np.floor((T_high - T_low) / T_step)) + 1
        self.data.bc = np.empty((numTs, b_vals.size))
        self.data.hw = np.empty((numTs, b_vals.size))

        # Execute code and fill bc and hw consecutively
        for ib, bg in enumerate(b_vals):
            bprimecmd = "bprime -T " + args["temprange"] + " -P " + str(args["p"]) + " -b " + str(bg) + " -m " + str(
                xmlfilename) + " -bl " + \
                        str(args["planet"]) + " -py " + str(gasname)

            expDYLD = 'export DYLD_LIBRARY_PATH=$MPP_DIRECTORY/install/lib:$DYLD_LIBRARY_PATH\n'
            h = StringIO(os.popen(expDYLD + bprimecmd).read())  # Execute mppequil command

            # Read table
            bhtab = pd.read_table(h, header=0, delim_whitespace=True, usecols=[0, 1, 2])

            self.data.bc[:, ib] = bhtab['B\'c'].values
            self.data.hw[:, ib] = bhtab['hw[MJ/kg]'].values * 1e6

        self.data.Tforbprime = bhtab['Tw[K]'].values
        self.data.bg = b_vals

        # Calculate interpolation functions
        self.bc = ip.interp2d(self.data.bg[:-1], self.data.Tforbprime, self.data.bc[:, :-1])
        self.hw = ip.interp2d(self.data.bg[:-1], self.data.Tforbprime, self.data.hw[:, :-1])

        # Calculate gradient functions
        self.dbcdT = lambda bg, T, tol=1.0e-8: (self.bc(bg, T + tol) - self.bc(bg, T - tol)) / (2 * tol)
        self.dbcdbg = lambda bg, T, tol=1.0e-8: (self.bc(bg + tol, T) - self.bc(bg - tol, T)) / (2 * tol)
        self.dhwdT = lambda bg, T, tol=1.0e-8: (self.hw(bg, T + tol) - self.hw(bg, T - tol)) / (2 * tol)
        self.dhwdbg = lambda bg, T, tol=1.0e-8: (self.hw(bg + tol, T) - self.hw(bg - tol, T)) / (2 * tol)

    def readBPrimes(self, args):
        """
        reads bprime table from csv file

        Parameters
        ----------
        args : dict
            dictionary of input arguments
            

        Returns
        -------

        """

        csvfile = args["bprime"]

        with open(csvfile) as f:
            data = pd.read_csv(f, sep=';', decimal='.', header=0, index_col=None)

        data = data.apply(pd.to_numeric, errors='coerce').dropna().to_numpy()

        # Get indices of time (where time changes first)
        ibgs = np.vstack((0, np.argwhere(np.diff(data[:, 0]) != 0) + 1)).flatten()
        self.data.bg = data[ibgs, 0]
        self.data.Tforbprime = data[0:ibgs[1], 1]

        self.data.bc = data[:, 2].reshape(len(self.data.bg), len(self.data.Tforbprime)).transpose()
        self.data.hw = data[:, 3].reshape(len(self.data.bg), len(self.data.Tforbprime)).transpose()

        # Calculate interpolation functions
        self.bc = ip.interp2d(self.data.bg[:-1], self.data.Tforbprime, self.data.bc[:, :-1])
        self.hw = ip.interp2d(self.data.bg[:-1], self.data.Tforbprime, self.data.hw[:, :-1])

        # Calculate gradient functions
        self.dbcdT = lambda bg, T, tol=1.0e-8: (self.bc(bg, T + tol) - self.bc(bg, T - tol)) / (2 * tol)
        self.dbcdbg = lambda bg, T, tol=1.0e-8: (self.bc(bg + tol, T) - self.bc(bg - tol, T)) / (2 * tol)
        self.dhwdT = lambda bg, T, tol=1.0e-8: (self.hw(bg, T + tol) - self.hw(bg, T - tol)) / (2 * tol)
        self.dhwdbg = lambda bg, T, tol=1.0e-8: (self.hw(bg + tol, T) - self.hw(bg - tol, T)) / (2 * tol)

    def plotBc(self):
        """plots Bc over T table with Bg as parameter"""
        for ib, bg in enumerate(self.data.bg):
            plt.semilogy(self.data.Tforbprime, self.data.bc[:, ib], label='%.2g' % bg)

        plt.xlabel('T [K]')
        plt.ylabel('B\'_c [-]')
        plt.grid(axis='x', which='major')
        plt.grid(axis='y', which='both')
        plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0, title='B\'_g')
        plt.subplots_adjust(right=0.7)
        plt.show()

    def plot_hw(self):
        """plots wall enthalpy hw over T with Bg as parameter
        """
        for ib, bg in enumerate(self.data.bg):
            plt.plot(self.data.Tforbprime, self.data.hw[:, ib], label='%.2g' % bg)
        plt.xlabel('T [K]')
        plt.ylabel('h_w [J/kg]')
        plt.grid(axis='x', which='major')
        plt.grid(axis='y', which='both')
        plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0, title='B\'_g')
        plt.subplots_adjust(right=0.7)
        plt.show()

    def storeVariables(self, args):
        """stores pressure and atmosphere values

        Parameters
        ----------
        args : dict
            dictionary of arguments

        Returns
        -------

        """
        self.pressure = float(args["p"])
        self.atmosphere = args["planet"]

    def calculateEnergies(self):
        """shifts virgin and char internal energies so that heat of formations are correct
         with respect to those calculated by Mutation++"""

        # Calculate shifts at Tref
        Tref = self.data.Tref
        self.virgin.eshift_Tref = self.virgin.data.hf - self.gas.data.hf + self.gas.h(Tref)
        self.char.eshift_Tref = self.char.data.hf - self.gas.data.hf + self.gas.h(Tref)

        # Calculate shift of antiderivative at T=0
        self.virgin.eshift_0 = self.virgin.eshift_Tref - self.virgin.cp.integral(0, Tref)
        self.char.eshift_0 = self.char.eshift_Tref - self.char.cp.integral(0, Tref)

        # Calculated unshifted energies
        self.virgin.e_unshifted = self.virgin.cp.antiderivative(1)
        self.char.e_unshifted = self.char.cp.antiderivative(1)

        # Calculate shifted energies
        self.virgin.e = lambda T: self.virgin.e_unshifted(T) + self.virgin.eshift_0
        self.char.e = lambda T: self.char.e_unshifted(T) + self.char.eshift_0

        # Calculate energies
        self.e = lambda T, wv: wv * self.virgin.e(T) + (1 - wv) * self.char.e(T)

    def calculateHatmo(self, args):
        """
        calculates atmospheric enthalpy

        Parameters
        ----------
        args : dict
            input arguments
            

        Returns
        -------

        """

        print("Calculating enthalpy of atmosphere...")

        # Construct Mutation++ mixture file
        xmldata = Path(hotstarship_dir, "Templates/mutationpp_mixtures_atmosphere.xml").read_text()
        xmldata = xmldata.replace("!elems", self.atmoelems)
        xmldata = xmldata.replace("!planet", self.atmosphere)
        xmlfilename = Path(args["input_dir"], "mutationpp_atmo_" + args["input_dir"].name + ".xml")
        with open(xmlfilename, 'w') as xmlfile:
            xmlfile.write(xmldata)

        # Construct mppequil command
        mppcmd = "mppequil -T " + args["temprange"] + " -P " + str(args["p"]) + " -m 0,10 " + str(xmlfilename)

        expDYLD = 'export DYLD_LIBRARY_PATH=$MPP_DIRECTORY/install/lib:$DYLD_LIBRARY_PATH\n'
        h = StringIO(os.popen(expDYLD + mppcmd).read())  # Execute mppequil command

        # Read table
        htab = pd.read_table(h, header=0, delim_whitespace=True)

        # Calculate enthalpy
        self.hatmo = constructLinearSpline(x=htab.values[:, 0], y=htab.values[:, 1])


def constructLinearSpline(x, y):
    """constructs a piecewise linear spline through data

    Parameters
    ----------
    x : np.ndarray
        x data as array (variable)
    y : np.ndarray
        y data as array (dependent)

    Returns
    -------
    type : ip.UnivariateSpline
        spline that is equal at nodes, and linear in-between

    """
    if len(x) != len(y):
        raise ValueError("Arrays should have same length.")
    if len(x) == 1:
        x = np.concatenate((x, x * 1.01))
        y = np.repeat(y, 2)

    y = np.hstack((y[0], y, y[-1]))
    x = np.hstack((0, x, 1e4))

    spline = ip.UnivariateSpline(x=x, y=y, k=1, s=0, ext='const')
    return spline


def constructE(Tforcp, cp, shift=0):
    """
    depreceated

    Parameters
    ----------
    Tforcp :
        
    cp :
        
    shift :
         (Default value = 0)

    Returns
    -------

    """
    if len(cp) == 1:
        cp = np.repeat(cp, 2)
        Tforcp = np.hstack((Tforcp, Tforcp * 2))
    elif len(cp) == 2:
        cp = np.array((cp[0], np.mean(cp), cp[1]))
        Tforcp = np.array((Tforcp[0], np.mean(Tforcp), Tforcp[1]))

    cp = np.hstack((cp[0], cp, cp[-1]))
    Tforcp = np.hstack((0, Tforcp, 1e4))

    e = cumtrapz(y=cp, x=Tforcp, initial=0)
    k = 1 if len(e) == 2 else 2
    eQuad = ip.UnivariateSpline(x=Tforcp, y=e + shift, k=k, ext='extrapolate')
    return e, eQuad


def dropnafromboth(x):
    """removes NaN rows and columns from pandas DataFrame

    Parameters
    ----------
    x : pandas.core.frame.DataFrame
        cleared DataFrame

    Returns
    -------
    x : pandas.core.frame.DataFrame
        cleared DataFrame

    """
    if type(x) is pd.DataFrame:
        x = x.dropna(how='all', axis=0)
        x = x.dropna(how='all', axis=1)
        return x
    else:
        raise TypeError


def checkForNonSI(data, file):
    """checks whether the user provided data might contain any non-SI units (safety measure)

    Parameters
    ----------
    data : pandas.core.frame.DataFrame
        pandas dataframe to be checked
    file : str
        file name used to issue warning

    Returns
    -------

    """
    # Concat row and column names
    names = '\t'.join(data.columns.values.astype(str)).upper() + '\t' + '\t'.join(data.index.values.astype(str)).upper()

    # Detect hits
    hitlocs = [list(re.finditer(nonsi, names)) for nonsi in nonsiunits]

    # Check if empty lists (= no hits)
    foundnonsi = [hitloc != [] for hitloc in hitlocs]

    if not any(foundnonsi):
        return  # if not hits, return
    else:

        nonsiconfirmed = False  # Flag variable

        # Check occurences of whitelisted words and flatten the list
        whitelocs = [list(re.finditer(white, names)) for white in whitelist]
        flatten = lambda l: [item for sublist in l for item in sublist]
        whitelocs = flatten(whitelocs)

        # For forbidden word check whether it is part of a whitelisted word
        for hitloc in list(compress(hitlocs, foundnonsi)):
            for match in hitloc:
                issubstring = [match.span()[0] >= whiteloc.span()[0] and match.span()[1] <= whiteloc.span()[1]
                               for whiteloc in whitelocs]
                if not any(issubstring):
                    nonsiconfirmed = True  # The forbidden word is not part of whitelisted word, set flag variable
                    break

            if nonsiconfirmed:
                warnings.warn("You might be using non-SI units in %s" % file, UserWarning)  # Issue Warning
                break


def handleArguments():
    """adds an argument parser and stores passed information

    Parameters
    ----------

    Returns
    -------
    args : dict
        arguments in dictionary form

    """
    # Create argument paser with respective options
    parser = argparse.ArgumentParser(description='Create material properties file (.matp) '
                                                 'for material calculation.')

    parser.add_argument('-i', '--input', action='store', dest='input_dir',
                        help="Input directory with material information (default current directory)")
    parser.add_argument('-o', '--output', action='store', dest='output_file',
                        help="Output file to write to (default <name of input directory>.matp")
    parser.add_argument('-a', '--ablative', action='store_true', dest='ablative',
                        help="Specify if material is ablative.", default=True)
    parser.add_argument('-na', '--non-ablative', action='store_false', dest='ablative',
                        help="Specify if material is non-ablative.")
    parser.add_argument('-T', '--temperature', action='store', dest='temprange',
                        help="Temperature range for bprime determination in K \"T1:dT:T2\" "
                             "or simply T (default = 300:100:6000 K)",
                        default='300:100:6000')
    parser.add_argument('-p', '--pressure', action='store', dest='p',
                        help="pressure in Pa for bprime determination (default = 101325 Pa (1 atm))", default='101325')
    parser.add_argument('-bg', '--gasblow', action='store', dest='bg',
                        help="pyrolysis gas blowing rate in \"b1:dB_order:b2\" (default 0.01:0.3333:100)",
                        default='0.01:0.3333:100')
    parser.add_argument('--planet', action='store', dest='planet',
                        help="planet whose atmospheric data is used (default Earth)", default="Earth")
    parser.add_argument('--bprime', action='store', dest='bprime',
                        help='Bprime table file to read (optional)', default=None)
    parser.add_argument('--hgas', action='store', dest='hgas',
                        help='Gas enthalpy values to read (optional)', default=None)
    args = vars(parser.parse_args())

    args["corrugated"] = False
    args["corrugated_vals"] = None

    args = checkInput(args)

    return args


def checkInput(args):
    """checks and corrects the input to material creation

    Parameters
    ----------
    args : dict
        dictionary of arguments

    Returns
    -------
    args : dict
        corrected dictionary of arguments

    """

    if not args["corrugated"]:
        # Construct input directory path
        if not args["input_dir"] or args["input_dir"] == ".":
            args["input_dir"] = Path.cwd()
        else:
            args["input_dir"] = input.find_existing(args["input_dir"], "material")

        # Construct output file path
        if not args["output_file"]:
            args["output_file"] = Path.joinpath(args["input_dir"], args["input_dir"].name + ".matp")
        else:
            args["output_file"] = Path.joinpath(Path.cwd(), args["output_file"])

    # Check whether pressure is in correct format
    try:
        float(args['p'])
    except ValueError:
        raise IOError("Check pressure input. Got %s" % args['p'])

    # Check if valid atmosphere
    if args["planet"] not in ('Earth', 'Mars', 'Venus'):
        raise IOError("Unimplemented planet %s" % args["planet"])

    # Construct bprime and hgas path
    if args["bprime"] is not None:
        args["bprime"] = input.find_existing(args["bprime"])
    if args["hgas"] is not None:
        args["hgas"] = input.find_existing(args["hgas"])


    return args


def createMaterial(inputdir, outfile=None, ablative=True, corrugated=False, Trange="300:100:6000", pressure=101325,
                   bg="0.01:0.3333:100",
                   atmosphere="Earth", corrugated_vals=None, bprime=None, hgas=None):
    """creates a Material class object that holds various thermophysical properties

    Parameters
    ----------
    corrugated_vals :
        saves information about corrugated layer properties (Default value = None)
    inputdir :
        input directory
    outfile :
        output file (Default value = None)
    ablative :
        ablative material flag (Default value = True)
    Trange :
        range of temperature for bprime calculation in K (format: min:step:max) (Default value = "300:100:6000")
    pressure :
        pressure for bprime calculation in Pa (Default value = 101325)
    bg :
        range of non-dimensional gas-blowing rate (format: min:orderOfMagnitudePerStep:max) (Default value = "0.01:0.3333:100")
    atmosphere :
        string that specifies the atmosphere (Default value = "Earth")
    corrugated :
         (Default value = False)
    bprime :
         (Default value = None)
    hgas :
         (Default value = None)

    Returns
    -------
    mat : material.Material
        Material instance

    """
    args = {}
    args["input_dir"] = inputdir
    args["output_file"] = outfile
    args["ablative"] = ablative
    args["corrugated"] = corrugated
    args["corrugated_vals"] = corrugated_vals
    args["temprange"] = Trange
    args["p"] = pressure
    args["bg"] = bg
    args["planet"] = atmosphere
    args["bprime"] = bprime
    args["hgas"] = hgas

    args = checkInput(args)

    # Read and store material information
    if args["ablative"]:
        material = AblativeMaterial(args)
    elif args["corrugated"]:
        material = CorrugatedMaterial(args)
    else:
        material = NonAblativeMaterial(args)

    # Write to .matp file
    if type(material) is not CorrugatedMaterial:
        with open(args["output_file"], 'wb') as outfile:
            # save using dill, recurse=True necessary for enabling pickling lambdify function from sympy
            dill.dump(material, outfile, recurse=True)

    return material


if __name__ == "__main__":

    # Get arguments
    args = handleArguments()

    # Read and store material information
    material = AblativeMaterial(args) if args["ablative"] else NonAblativeMaterial(args)

    print("Writing .matp file to %s..." % args["output_file"])

    # Write to .matp file
    with open(args["output_file"], 'wb') as outfile:
        # save using dill, recurse=True necessary for enabling pickling lambdify function from sympy
        dill.dump(material, outfile, recurse=True)

    print("Wrote material.")
