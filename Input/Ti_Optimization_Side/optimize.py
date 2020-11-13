from pathlib import Path
import os
import hotstarship
from scipy.optimize import Bounds, NonlinearConstraint, minimize
import numpy as np
import math
import sympy as sp
from sympy import hessian

# Variables
# 0: t_f: top face sheet thickness [m]
# 1: t_c: core thickness [m]
# 2: t_b: bottom face sheet thickness [m]
# 3: t_w: web thickness [m]
# 4: Theta: corrugation angle [10^4 °] (for scaling reasons)
# 5: p: half cell length [m]
x0 = [2e-3, 50e-3, 2e-3, 2e-3, 0.0045, 50e-3]

# Function tolerance at which to terminate
tolerance = 1e-6

# Lower and upper bounds
lb = [0.5e-3, 2e-3, 0.5e-3, 0.5e-3, 10e-4, 3e-3]
ub = [20e-3, 100e-3, 20e-3, 20e-3, 60e-4, 100e-3]

# Constraints
# 0: back face temperature
# 1: maximum temperature
max_back_T = 373.15
max_T = 1000

# Material densities
rho_sheet = 4420 # for Ti-6Al-4V
rho_core = 200   # for Pyrogel XTE


class Functions:
    def __init__(self):
        vars = sp.var('t_f t_c t_b t_w Theta p')
        mass = rho_sheet * t_f + \
               (rho_sheet * t_w + rho_core * (p * sp.sin(sp.rad(Theta * 1e4)) - t_w)) / \
               (p * sp.sin(sp.rad(Theta * 1e4))) * t_c + \
               rho_sheet * t_b

        # Function
        f = sp.Matrix([mass])
        X = sp.Matrix([t_f, t_c, t_b, t_w, Theta, p])
        self.f_lam = sp.lambdify(vars, f, 'numpy')

        # Jacobian
        J = f.jacobian(X)
        self.J_lam = sp.lambdify(vars, J, 'numpy')

        # Hessian
        H = hessian(f, vars)
        self.H_lam = sp.lambdify(vars, H, 'numpy')

    def J(self, args):
        print("Running Jacobian fun with: " + str(args))
        return self.J_lam(*args)

    def H(self, args):
        print("Running Hessian  fun with: " + str(args))
        return self.H_lam(*args)

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
    xmldata = xmldata.replace("**t_f**", str(x[0]))
    xmldata = xmldata.replace("**firstcell_f**", str(x[0] / 100))
    xmldata = xmldata.replace("**t_c**", str(x[1]))
    xmldata = xmldata.replace("**t_b**", str(x[2]))
    xmldata = xmldata.replace("**t_w**", str(x[3]))
    xmldata = xmldata.replace("**Theta**", str(x[4] * 1e4))
    xmldata = xmldata.replace("**p**", str(x[5]))

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
            max_T - sr.get_max_T()]

    print("Back face temperature constraint: %.2f K difference" % -cons[0])
    print("Max       temperature constraint: %.2f K difference" % -cons[1])

    return cons


if __name__ == "__main__":

    # Set up target function, Jacobian and Hessian
    funs = Functions()

    # Set bounds for thickness
    bounds = Bounds(lb=lb, ub=ub, keep_feasible=True)

    # Set constraints for temperature and remaining thickness
    ineq_cons = {'type': 'ineq', 'fun': constraint_fun}

    # Start optimization procedure
    res = minimize(funs.f, x0, jac=funs.J, hess=funs.H, method='SLSQP', constraints=[ineq_cons],
                   options={'ftol': tolerance, 'eps': tolerance}, bounds=bounds)

    # Print result
    print(res)
