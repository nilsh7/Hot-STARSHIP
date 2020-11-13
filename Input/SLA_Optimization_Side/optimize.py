from pathlib import Path
import os
import hotstarship
from scipy.optimize import Bounds, NonlinearConstraint, minimize
import numpy as np
import math
import sympy as sp
from sympy import hessian

# Variables
# 0: t: ablative material thickness [m]
x0 = [50e-3]

# Function tolerance at which to terminate
tolerance = 1e-6

# Lower and upper bounds
lb = [10.0e-3]
ub = [200e-3]

# Constraints
# 0: back face temperature
# 1: remaining thickness in percent
max_back_T = 373.15
remaining_thickness_min = 0.1

# Material densities
rho = 232.29 # for SLA-561


class Functions:
    def __init__(self):
        vars = sp.var('t')
        mass = rho * t

        # Function
        f = sp.Matrix([mass])
        X = sp.Matrix([t])
        self.f_lam = sp.lambdify(vars, f, 'numpy')

        # Jacobian
        J = f.jacobian(X)
        self.J_lam = sp.lambdify(vars, J, 'numpy')

        # Hessian
        # H = hessian(f, vars)
        # self.H_lam = sp.lambdify(vars, H, 'numpy')

    def J(self, args):
        print("Running Jacobian fun with: " + str(args))
        return self.J_lam(*args)

    #def H(self, args):
    #    print("Running Hessian  fun with: " + str(args))
    #    return self.H_lam(*args)

    def f(self, args):
        print("Running target   fun with: " + str(args))
        mass = self.f_lam(*args)[0][0]
        print("Mass is %.2f kg/m^2." % mass)
        return mass


def constraint_fun(x):
    print("Running constr fun with: " + str(x))

    # Generate input file
    script_dir = os.path.dirname(os.path.realpath(__file__))
    xmldata = Path(script_dir, "input_Template.xml").read_text()
    xmldata = xmldata.replace("**t**", str(x[0]))
    xmldata = xmldata.replace("**firstcell**", str(x[0] / 100))

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
    cons = [max_back_T - sr.get_max_back_T(),
            sr.get_remaining_thickness()/x[0] - remaining_thickness_min]

    print("Back face temperature constraint: %.2f K difference" % -cons[0])
    print("Remaining thickness   constraint: {:.1%} difference".format(cons[1]))

    return cons


if __name__ == "__main__":

    # Set up target function, Jacobian and Hessian
    funs = Functions()

    # Set bounds for thickness
    bounds = Bounds(lb=lb, ub=ub, keep_feasible=True)

    # Set constraints for temperature and remaining thickness
    ineq_cons = {'type': 'ineq', 'fun': constraint_fun}

    # Start optimization procedure
    res = minimize(funs.f, x0, jac=funs.J, method='SLSQP', constraints=[ineq_cons],
                   options={'ftol': tolerance, 'eps': tolerance}, bounds=bounds)

    # Print result
    print(res)
