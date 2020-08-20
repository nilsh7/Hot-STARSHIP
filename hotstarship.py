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

    parser.add_argument("input_file", help="input file to read from (xml format)")

    parser.add_argument("output_file", help="output file to write to (csv format)", nargs='?',
                        default=None)

    parser.add_argument('--force', action='store_true', dest='force_write',
                        help="force to overwrite solution file if it already exists",
                        default=False)

    args = vars(parser.parse_args())

    return args


def hotstarship(args):

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

    # Initialize output file and write initial distribution
    solwrite = output.SolutionWriter(args["output_file"], layers, Tmap, inputvars, args["force_write"])
    solwrite.write(inputvars.tStart, layers, Tnu, rhonu, Tmap, rhomap, mgas)

    # Initialize deltaTn guess with zero change
    deltaTn = np.zeros(len(Tnu))

    # Calculate smoothing function for Tmin
    if layers[0].ablative:
        calculateSmoothStep()

    last_step = len(inputvars.ts) - 1
    for it, t in enumerate(inputvars.ts):

        # Print some information about current time step
        print("\n+++ New time step: t = %.4f secs +++" % t)
        inputvars.tDelta = inputvars.ts[it] - inputvars.ts[it-1]\
            if it != 0 else inputvars.ts[it] - inputvars.tStart

        # Copy values of previous time step for time dependent function values
        Tn, rhon, rhoin = Tnu.copy(), rhonu.copy(), rhoimu.copy()

        # Save information about grid of previous time step
        gridpre_f = dill.copy(layers[0].grid) if layers[0].ablative else layers[0].grid
        wvpre_f = layers[0].wv.copy() if layers[0].ablative else layers[0].wv

        # Set global iteration counter
        iteration = 0

        # Initial guess based on difference at previous time step
        Tnu += deltaTn
        #if layers[0].ablative:
        #    rhonu, rhoimu, mgas = updateRho(layers[0], rhoimu, rhoin, rhonu, rhomap, Tnu, Tmap, inputvars.tDelta)

        while True:

            # Increment lower level iteration counter
            iteration += 1

            # Update rho
            if layers[0].ablative:
                rhomu_m1 = rhonu.copy()
                rhonu, rhoimu, mgas = updateRho(layers[0], rhoimu, rhoin, rhonu, rhomap, Tnu, Tmap, inputvars.tDelta)
                deltaRhon = rhonu - rhon

            # Assemble Jacobian and function vector
            J, f = assembleT(layers, gridpre_f, wvpre_f, Tmap, Tnu, Tn, rhomap, rhonu, rhon, mgas, t, inputvars.tDelta, inputvars)

            # Solve for difference
            dT = spsolve(J, -f)

            # Apply difference
            Tnu += dT

            # Update grid when using ablative calculation
            if layers[0].ablative:
                # On very first iteration, update using actual sdot value
                if iteration == 1:
                    sdot = Tnu[Tmap["sdot"]]
                    layers[0].grid.updateZ(delta_s=sdot * inputvars.tDelta)
                # otherwise, update using difference between current and last iteration
                else:
                    delta_sdot = dT[0]
                    layers[0].grid.updateZ(delta_s=delta_sdot * inputvars.tDelta)

            # Stop iterating if convergence in T (and possibly sdot) has been reached
            nonzero = Tnu > 1.0e-7
            if np.max(np.abs(dT[nonzero]/Tnu[nonzero])) < 1.0e-8: #and iteration > 2:
            #if np.linalg.norm(dT/Tnu) < 1.0e-5:
                print("Completed after %i iterations." % iteration)
                deltaTn = Tnu - Tn
                if ((it+1) % inputvars.write_step) == 0 or (it == last_step):
                    solwrite.write(t, layers, Tnu, rhonu, Tmap, rhomap, mgas)
                break

        if layers[0].ablative:
            if (layers[0].grid.s + Tnu[Tmap["sdot"]] * inputvars.tDelta) / layers[0].grid.length0 > 0.97:
                print("Material recession will probably exceed 97% in the next time step.\n"
                      "Terminating here.")
                solwrite.write(t, layers, Tnu, rhonu, Tmap, rhomap, mgas)
                return False

    # Option to compare test case to analytical profile
    # comp.compareToAnalytical(t, Tnu, inputvars)

    # Plot output
    #output.plotT(layers, Tnu, Tmap, t, inputvars)
    #output.plotBeta(layers, rhonu, rhomap, t)

    # End has been reached (final statement for debugging purposes)
    return True


def get_mass(args):

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

    # Get mass
    mass = output.calc_weight(layers, rhonu, rhomap)

    return mass


if __name__ == "__main__":

    # Save input arguments
    args = handleArguments()

    hotstarship(args)
