import argparse
import input
import numpy as np
from assembly import *
import grid
import dill
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import Testing.comparison as comp
import output

def handleArguments():
    """
handles arguments passed to Hot-STARSHIP
    :return: dictionary of arguments
    """
    # Create argument paser with respective options
    parser = argparse.ArgumentParser(description='Pass an input.xml file for calculation with Hot-STARSHIP')

    parser.add_argument("input_file")

    args = vars(parser.parse_args())

    return args


def savePreValues(layers):
    layerspre = dill.copy(layers)
    return layerspre


if __name__ == "__main__":
    # Save input arguments
    args = handleArguments()

    # Read input file
    print("Reading input file %s..." % args["input_file"])
    inputvars = input.Input(args["input_file"])
    layers = inputvars.layers  # Extract layers

    # Generate grids
    layers = grid.generateGrids(layers)

    # Add variables
    layers = addVariables(layers)

    # Create vectors to store unknowns in and dictionary from layer to indices
    Tnu, rhonu, Tmap, rhomap = createUnknownVectors(layers)

    # Initialize variables to be solved
    Tnu, rhonu, rhoimu, mgas = init_T_rho(Tnu, rhonu, Tmap, rhomap, layers, inputvars)

    #Tnu = np.linspace(250, 750, len(Tnu))  # for debugging purposes
    deltaTn = np.linspace(0.0, 0.0, len(Tnu))

    for it, t in enumerate(np.arange(inputvars.tStart, inputvars.tEnd + 1e-5, inputvars.tDelta)):
        print("+++ New time step: t = %.4f secs +++" % t)

        Tn, rhon, rhoin = Tnu.copy(), rhonu.copy(), rhoimu.copy()
        #Tn[1:] += 50*np.arange(len(Tn)-1)  # for debugging

        globiteration = 0

        layerspre = savePreValues(layers)

        while True:

            globiteration += 1
            iteration = 0
            Tnu += deltaTn  # Initial guess based on previous difference

            while True:

                #break  # TODO: for debugging purposes with ablative material
                rhonu, rhoimup1, mgas = updateRho(layers[0], rhoimu, rhoin, Tnu, Tmap, inputvars.tDelta)  # TODO: for debugging purposes with ablative material

                iteration += 1

                J, f = assembleT(layers, layerspre, Tmap, Tnu, Tn, rhomap, rhonu, rhon, mgas, inputvars.tDelta, inputvars)

                f[0] += -7.5e5

                #f[1] += -7.5e5
                #Jd = J.toarray()
                #dT = np.zeros(f.shape)
                #dT[1:] = np.linalg.solve(Jd[1:, 1:], -f[1:])

                dT = spsolve(J, -f)

                Tnu += dT

                if np.linalg.norm(dT/Tnu) < 1.0e-8: #and iteration > 2:
                    print("T determination completed after %i iterations." % iteration)
                    deltaTn = Tnu - Tn
                    break

            if not layers[0].ablative:
                break
            else:
                rhonu, rhoimup1, mgas = updateRho(layers[0], rhoimu, rhoin, Tnu, Tmap, inputvars.tDelta)
                if np.linalg.norm((rhoimup1-rhoimu)/rhoimu) < 1.0e-8:
                    print("Time step completed after %i iterations" % globiteration)
                    break

    #comp.compareToAnalytical(t, Tnu, inputvars)

    output.plotT(layers, Tnu, Tmap, t)

    #plt.plot(layers[0].grid.zj, Tnu)
    #plt.show()

    aaa = 1
