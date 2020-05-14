from pathlib import Path
import pandas as pd
import warnings
from numpy.polynomial.polynomial import Polynomial
import numpy as np
import sympy as sp
from sympy import Piecewise, integrate
from sympy.utilities.lambdify import lambdify
import argparse
import re
from itertools import compress
import dill  # for pickling lambdas
import os
from io import StringIO

# previously used packages: csv, os, glob, plt, json, jsonpickle, pickle

nonsiunits = [unit.upper() for unit in ("ft", "°R", "BTU", "lb", "°F", "°C", "cal", "hr")]
whitelist = [word.upper() for word in ("Scaled",)]

converters = {"-": 0, "#WERT!": 0}

convertf = lambda st: "0,0" if st in ("-", "#WERT!") else st


class Material:
    def __init__(self):
        self.data = Data()
        self.T = sp.symbols("T")

    def readFile(self):
        pass  # to be intitiated in Ablative or NonAblativeMaterial

    def calculateVariables(self):
        pass  # to be intitiated in Ablative or NonAblativeMaterial

    def readData(self):

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
                    self.readFile(iDir, globdir, csv_files)


class State:
    def __init__(self):
        self.data = Data()


class Data:
    def __init__(self):
        pass


class NonAblativeMaterial(Material):
    def __init__(self):
        Material.__init__(self)

        # List of necessary directories
        self.necessarydirs = [Path(d) for d in ["cp", "eps", "k"]]

        self.readData()

        self.calculateVariables()

    def readFile(self, iDir, globdir, csv_files):

        # Determine csv file
        csv_file = Path.joinpath(globdir, csv_files[0])

        # Open file
        with open(csv_file) as f:
            data = pd.read_csv(f, sep=';', decimal=',')

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

    def calculateVariables(self):

        # cp
        self.data.cpPiece = constructPiecewise(self.data.Tforcp, self.data.cp, self.T)
        self.cp = lambdify(self.T, self.data.cpPiece, 'numpy')

        # k
        self.data.kPiece = constructPiecewise(self.data.Tfork, self.data.k, self.T)
        self.k = lambdify(self.T, self.data.kPiece, 'numpy')

        # eps
        self.data.epsPiece = constructPiecewise(self.data.Tforeps, self.data.eps, self.T)
        self.eps = lambdify(self.T, self.data.epsPiece, 'numpy')


class AblativeMaterial(Material):
    def __init__(self):
        Material.__init__(self)
        self.virgin = State()
        self.char = State()
        self.gas = State()

        self.beta = sp.symbols("beta")

        self.necessarydirs = [Path(d) for d in ["Virgin/cp", "Virgin/eps", "Virgin/k", "Virgin/comp",
                                                "Char/cp", "Char/eps", "Char/k",
                                                "Gas/h",
                                                "Combined"]]

        self.readData()

        self.calculateVariables()

        self.calculateAblativeProperties()

    def readFile(self, iDir, globdir, csv_files):
        # Open files and read data
        global data
        if iDir != 8:
            csv_file = Path.joinpath(globdir, csv_files[0])

            head, index = (None, 0) if iDir==3 else (0, None)


            # Open file
            with open(csv_file) as f:
                data = pd.read_csv(f, sep=';', decimal=',', header=head, index_col=index)

            data = dropnafromboth(data)

            checkForNonSI(data, csv_file)

        else:
            # Read heats of formation
            hof_file = Path.joinpath(globdir, "Heats_of_Formation.csv")

            # Open file
            with open(hof_file) as f:
                hof_data = pd.read_csv(f, sep=';', decimal=',', header=None)

            hof_data = dropnafromboth(hof_data)

            checkForNonSI(hof_data, hof_file)

            # Read Decomposition Kinetics
            dec_file = Path.joinpath(globdir, "Decomposition_Kinetics.csv")

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
            #self.virgin.data.comp = {"C": data.values[0],
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
            self.char.data.k = data.values[:, 0]
        elif iDir == 7:
            self.gas.data.Tforh = data.values[:, 0]
            self.gas.data.h = data.values[:, 1]
        elif iDir == 8:

            # Heats of formation
            self.data.Tref = hof_data.values[0, 1]
            self.virgin.data.hf = hof_data.values[1, 1]
            self.char.data.hf = hof_data.values[2, 1]
            self.gas.data.hf = hof_data.values[3, 1]

            # Decomposition kinetics
            self.data.nDecomp = dec_data.columns.values.size - 1
            self.data.virginRho0 = dec_data.values[0, :]
            self.data.charRho0 = dec_data.values[1, :]
            self.data.kr = dec_data.values[2, :]
            self.data.nr = dec_data.values[3, :]
            self.data.Ei = dec_data.values[4, :]
            self.data.Tmin = dec_data.values[5, :]
            self.data.frac = dec_data.values[6, :]

    def calculateVariables(self):

        ### Virgin ###
        # cp
        self.virgin.data.cpPiece = constructPiecewise(self.virgin.data.Tforcp, self.virgin.data.cp, self.T)
        self.virgin.cp = lambdify(self.T, self.virgin.data.cpPiece, 'numpy')

        # k
        self.virgin.data.kPiece = constructPiecewise(self.virgin.data.Tfork, self.virgin.data.k, self.T)
        self.virgin.k = lambdify(self.T, self.virgin.data.kPiece, 'numpy')

        # eps
        self.virgin.data.epsPiece = constructPiecewise(self.virgin.data.Tforeps, self.virgin.data.eps, self.T)
        self.virgin.data.eps = lambdify(self.T, self.virgin.data.epsPiece, 'numpy')

        # e
        self.virgin.data.ePiece = self.virgin.data.hf + integrate(self.virgin.data.cpPiece,
                                                                  (self.T, self.data.Tref, self.T))
        self.virgin.data.eLam = lambdify(self.T, self.virgin.data.ePiece, 'numpy')
        self.virgin.e = lambda Ts: [self.virgin.data.eLam(T) for T in Ts]

        ### Char ###
        # cp
        self.char.data.cpPiece = constructPiecewise(self.char.data.Tforcp, self.char.data.cp, self.T)
        self.char.cp = lambdify(self.T, self.char.data.cpPiece, 'numpy')

        # k
        self.char.data.kPiece = constructPiecewise(self.char.data.Tfork, self.char.data.k, self.T)
        self.char.k = lambdify(self.T, self.char.data.kPiece, 'numpy')

        # eps
        self.char.data.epsPiece = constructPiecewise(self.char.data.Tforeps, self.char.data.eps, self.T)
        self.char.data.eps = lambdify(self.T, self.char.data.epsPiece, 'numpy')

        # e
        self.char.data.ePiece = self.char.data.hf + integrate(self.char.data.cpPiece,
                                                              (self.T, self.data.Tref, self.T))
        self.char.data.eLam = lambdify(self.T, self.char.data.ePiece, 'numpy')
        self.char.e = lambda Ts: [self.char.data.eLam(T) for T in Ts]

        ### Gas ###
        self.gas.data.hPiece = constructPiecewise(self.gas.data.Tforh, self.gas.data.h, self.T)
        self.gas.data.hPiece = self.gas.data.hPiece - self.gas.data.hPiece.evalf(subs={self.T: self.data.Tref}) + \
                               self.gas.data.hf
        self.gas.h = lambdify(self.T, self.gas.data.hPiece, 'numpy')

        ### Combined ###
        # cp
        self.data.cpPiece = (1 - self.beta) * self.virgin.data.cpPiece + self.beta * self.char.data.cpPiece
        self.cp = lambdify([self.T, self.beta], self.data.cpPiece, 'numpy')

        # k
        self.data.kPiece = (1 - self.beta) * self.virgin.data.kPiece + self.beta * self.char.data.kPiece
        self.k = lambdify([self.T, self.beta], self.data.kPiece, 'numpy')

        # eps
        self.data.epsPiece = (1 - self.beta) * self.virgin.data.epsPiece + self.beta * self.char.data.epsPiece
        self.eps = lambdify([self.T, self.beta], self.data.epsPiece, 'numpy')

        # e
        self.data.ePiece = (1 - self.beta) * self.virgin.data.ePiece + self.beta * self.char.data.ePiece
        self.e = lambdify([self.T, self.beta], self.data.ePiece, 'numpy')

    def calculateAblativeProperties(self):

        self.calculatePyroGasComposition()

    def calculatePyroGasComposition(self):

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

        # Construct Mutation++ mixture file
        xmldata = Path("Templates/mutationpp_mixtures.xml").read_text()

        xmldata = xmldata.replace("!name", args["input_dir"].name)  # Replace name input dir name
        moletxt = ""
        for index, row in a.iterrows():
            moletxt += str(row.name) + str(":") + str(row["pyromolefrac"]) + ", "
        xmldata = xmldata.replace("!gascomp", moletxt[:-2])  # Replace mole fractions

        xmlfilename = Path(args["input_dir"], "mutationpp_" + args["input_dir"].name + ".xml")
        with open(xmlfilename, 'w') as xmlfile:
            xmlfile.write(xmldata)

        # Construct mppequil command
        moletxt = ""
        for index, row in a.iterrows():
            moletxt += str(row.name) + str(":") + str(row["pyromolefrac"]) + ","
        mppcmd = "mppequil -T " + args["temprange"] + " -P " + args["prange"] + " -m 0,10 " + str(xmlfilename)

        expDYLD = 'export DYLD_LIBRARY_PATH=$MPP_DIRECTORY/install/lib:$DYLD_LIBRARY_PATH\n'
        h = StringIO(os.popen(expDYLD + mppcmd).read())  # Execute mppequil command

        # Read table
        htab = pd.read_table(h, header=0, delim_whitespace=True)

        # Calculate enthalpy
        self.gas.data.hPiece = constructPiecewise(htab.values[:, 0], htab.values[:, 1], self.T)
        self.gas.data.hPiece = self.gas.data.hPiece - self.gas.data.hPiece.evalf(subs={self.T: self.data.Tref}) + \
                               self.gas.data.hf
        self.gas.h = lambdify(self.T, self.gas.data.hPiece, 'numpy')


def constructPiecewise(x, y, symbol):
    sorted_order = np.argsort(x)
    x = x[sorted_order]
    y = y[sorted_order]

    start = ((y[+0], symbol < x[+0]),)
    end = ((y[-1], symbol >= x[-1]),)
    mid = ()
    for i in np.arange(np.size(x) - 1):
        mid += (((y[i + 1] - y[i]) / (x[i + 1] - x[i]) * (symbol - x[i]) + y[i], symbol <= x[i + 1]),)

    piecewisef = start + mid + end

    return Piecewise(*piecewisef)

def dropnafromboth(x):
    if type(x) is pd.DataFrame:
        x = x.dropna(how='all', axis=0)
        x = x.dropna(how='all', axis=1)
        return x
    else:
        raise TypeError

def createPoly(x, y, polydeg, symbol):
    # Calculate maximum possible degree if specified
    if polydeg == "max":
        polydeg = len(x) - 1

    coeffs = Polynomial.fit(x=x, y=y, deg=min(len(x) - 1, polydeg), window=np.array([np.min(x), np.max(x)]))
    poly = sp.Poly(np.flip(coeffs.coef), symbol)
    return poly


def checkForNonSI(data, file):
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
    # Create argument paser with respective options
    parser = argparse.ArgumentParser(description='Create material properties file (.matp) '
                                                 'for ablative material calculation.')

    parser.add_argument('-i', '--input', action='store', dest='input_dir',
                        help="Input directory with material information (default current directory)")
    parser.add_argument('-o', '--output', action='store', dest='output_file',
                        help="Output file to write to (default <name of input directory>.matp")
    parser.add_argument('-a', '--ablative', action='store_true', dest='ablative',
                        help="Specify if material is ablative.", default=True)
    parser.add_argument('-na', '--non-ablative', action='store_false', dest='ablative',
                        help="Specify if material is non-ablative.")
    parser.add_argument('-T', '--temperature', action='store', dest='temprange',
                        help="Temperature range in K \"T1:dT:T2\" or simply T (default = 200:100:3000 K)",
                        default='200:100:3000')
    parser.add_argument('-P', '--pressure', action='store', dest='prange',
                        help="pressure range in Pa \"P1:dP:P2\" or simply P (default = 1 atm)", default='1')
    args = vars(parser.parse_args())

    # Construct input directory path
    if not args["input_dir"]:
        args["input_dir"] = Path.cwd()
    else:
        args["input_dir"] = Path.joinpath(Path.cwd(), args["input_dir"])

    # Construct output file path
    if not args["output_file"]:
        args["output_file"] = Path.joinpath(args["input_dir"], args["input_dir"].name + ".matp")
    else:
        args["output_file"] = Path.joinpath(Path.cwd(), args["output_file"])

    return args


if __name__ == "__main__":
    # Get arguments
    args = handleArguments()

    # Read and store material information
    material = AblativeMaterial() if args["ablative"] else NonAblativeMaterial()

    # Write to .matp file
    with open(args["output_file"], 'wb') as outfile:
        # save using dill, recurse=True necessary for enabling pickling lambdify function from sympy
        dill.dump(material, outfile, recurse=True)
