import argparse
import input
import numpy as np
from assemblation import *
import grid

def handleArguments():
    # Create argument paser with respective options
    parser = argparse.ArgumentParser(description='Pass an input.xml file for calculation with Hot-STARSHIP')

    parser.add_argument("input_file")

    args = vars(parser.parse_args())

    return args


if __name__ == "__main__":
    # Save input arguments
    args = handleArguments()

    # Read input file
    print("Reading input file %s..." % args["input_file"])
    inputvars = input.Input(args["input_file"])
    layers = inputvars.layers  # Extract layers

    # Generate grids
    layers = grid.generateGrids(layers)

    # Create vectors to store unknowns in and dictionary from layer to indices
    Tn, rhon, Tmap, rhomap = createUnknownVectors(layers, inputvars.ablative)

    # Initialize variables to be solved
    Tn, rhon = init_T_rho(Tn, rhon, Tmap, rhomap, layers, inputvars)

    for it, t in enumerate(np.arange(inputvars.tStart, inputvars.tEnd + 1e-5, inputvars.tDelta)):
        print("+++ New time step: t = %.4f secs +++" % t)

        iteration = 0

        while True:
            iteration += 1

    aaa = 1
