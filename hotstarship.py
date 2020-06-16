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
    Tnu, rhonu, rhoimu, Tmap, rhomap = createUnknownVectors(layers)

    # Initialize variables to be solved
    Tnu, rhonu, rhoimu, mgas = init_T_rho(Tnu, rhonu, rhoimu, Tmap, rhomap, layers, inputvars)

    # Initialize deltaTn guess with zero change
    deltaTn = np.zeros(len(Tnu))

    for it, t in enumerate(np.arange(inputvars.tStart+inputvars.tDelta, inputvars.tEnd + 1e-5, inputvars.tDelta)):

        # Print some information about current time step
        print("+++ New time step: t = %.4f secs +++" % t)

        # Copy values of previous time step for time dependent function values
        Tn, rhon, rhoin = Tnu.copy(), rhonu.copy(), rhoimu.copy()

        # Save information about grid of previous time step
        layerspre = savePreValues(layers)

        # Set global iteration counter
        globiteration = 0

        # Initial guess based on difference at previous time step
        Tnu += deltaTn
        if layers[0].ablative:
            rhonu, rhoimu, mgas = updateRho(layers[0], rhoimu, rhoin, rhonu, rhomap, Tnu, Tmap, inputvars.tDelta)

        while True:

            # Increment higher level iteration counter, initialize lower level iteration counter
            globiteration += 1
            iteration = 0

            while True:

                # Increment lower level iteration counter
                iteration += 1

                # Assemble Jacobian and function vector
                J, f = assembleT(layers, layerspre, Tmap, Tnu, Tn, rhomap, rhonu, rhon, mgas, t, inputvars.tDelta, inputvars)

                # Solve for difference
                dT = spsolve(J, -f)

                # Apply difference
                Tnu += dT

                # Update grid when using ablative calculation
                if layers[0].ablative:
                    # On very first iteration, update using actual sdot value
                    if globiteration == 1 and iteration == 1:
                        sdot = Tnu[Tmap["sdot"]]
                        layers[0].grid.updateZ(delta_s=sdot * inputvars.tDelta)
                    # otherwise, update using difference between current and last iteration
                    else:
                        delta_sdot = dT[0]
                        layers[0].grid.updateZ(delta_s=delta_sdot * inputvars.tDelta)

                # Stop iterating if convergence in T (and possibly sdot) has been reached
                if np.max(np.abs(dT/Tnu)) < 1.0e-8: #and iteration > 2:
                #if np.linalg.norm(dT/Tnu) < 1.0e-5:
                    print("-> T determination completed after %i iterations." % iteration)
                    deltaTn = Tnu - Tn
                    break

            # If not dealing with an ablative case, go to next time step
            if not layers[0].ablative:
                break
            # else update nodal densities
            else:
                rhomu_m1 = rhonu.copy()
                rhonu, rhoimu, mgas = updateRho(layers[0], rhoimu, rhoin, rhonu, rhomap, Tnu, Tmap, inputvars.tDelta)
                if np.max(np.abs((rhonu-rhomu_m1)/rhomu_m1)) < 1.0e-8:
                    print("--> Time step completed after %i iterations." % globiteration)
                    deltaRhon = rhonu - rhon
                    break

    # Option to compare test case to analytical profile
    # comp.compareToAnalytical(t, Tnu, inputvars)

    # Plot output
    output.plotT(layers, Tnu, Tmap, t, inputvars)
    output.plotBeta(layers, rhonu, rhomap, t)

    # End has been reached (final statement for debugging purposes)
    end = True

    aaa = 1
