import argparse
import input
import numpy as np
from assembly import *
import grid
import dill
from scipy.sparse.linalg import spsolve

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
    for lay in layers:
        lay.pre = dill.copy(lay)


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
    Tnu, rhonu = init_T_rho(Tnu, rhonu, Tmap, rhomap, layers, inputvars)

    #Tnu = np.linspace(250, 750, len(Tnu))  # for debugging purposes
    deltaTn = np.linspace(1.0, 0, len(Tnu))

    for it, t in enumerate(np.arange(inputvars.tStart, inputvars.tEnd + 1e-5, inputvars.tDelta)):
        print("+++ New time step: t = %.4f secs +++" % t)

        Tn, rhon = Tnu.copy(), rhonu.copy()
        #Tn[1:] += 50*np.arange(len(Tn)-1)  # for debugging

        iteration = 0

        savePreValues(layers)
        Tnu += deltaTn  # Initial guess based on previous difference

        while True:

            iteration += 1

            J, f = assembleT(layers, Tmap, Tnu, Tn, rhomap, rhonu, rhon, inputvars.tDelta, inputvars)

            f[0] += 0

            dT = spsolve(J, -f)

            Tnu += dT

            if np.linalg.norm(dT/Tnu) < 1.0e-6:
                print("Completed after %i iterations." % iteration)
                deltaTn = Tnu - Tn
                break

    aaa = 1
