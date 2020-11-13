from pathlib import Path
import os
import hotstarship
from scipy.optimize import Bounds, NonlinearConstraint, minimize
import numpy as np
import math
import sympy as sp
from sympy import hessian
from scipy.optimize import newton, toms748

# Constraint: back face temperature
max_back_T = 373.15

# Temperature function
def temp_fun(x):
    print("Running temp fun with t = " + str(x*1e3) + " mm")

    # Generate input file
    script_dir = os.path.dirname(os.path.realpath(__file__))
    xmldata = Path(script_dir, "input_Template.xml").read_text()
    xmldata = xmldata.replace("**t_c**", str(x))

    # Save input file
    input_file = Path(script_dir, "input.xml")
    with open(input_file, 'w') as xmlfile:
        xmlfile.write(xmldata)

    # Generate input arguments
    output_file = str(Path(script_dir, "output.csv"))
    args = {"input_file": input_file,
            "output_file": output_file,
            "force_write": True}

    # Run Hot-STARSHIP
    valid = hotstarship.hotstarship(args)

    # Evaluate constraints
    sr = hotstarship.output.SolutionReader(output_file)
    root = max_back_T - sr.get_max_back_T()

    print("^-> Max back T: %.2f" % sr.get_max_back_T())

    return root


if __name__ == "__main__":

    # If you wish to save the iteration steps, run with:
    # python3 root_finding.py > log.txt
    #
    # Display iteration steps with:
    # less log.txt | grep "Running"


    # Start optimization
    tc_goal, res = toms748(temp_fun, a=0.005, b=0.1, xtol=0.1e-3, full_output=True)

    # Print final value
    print("Final value is: " + str(tc_goal))

    # Print result
    print(res)
