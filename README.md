# Hot-STructure and Ablative Reaction SHIeld Program
This Python program calculates the thermal response of ablative and non-ablative rocket or spacecraft heat shields. The code is a 1D finite volume method.

Some features include:
- ablative, non-ablative or corrugated sandwich materials
- temperature-dependent properties
- integration of generation of chemistry data by use of [Mutation++](https://github.com/mutationpp/Mutationpp)
- integrated plotting functions
- recovery enthalpy or heat flux boundary conditions
- only adiabatic back face boundary condition (so far)

The code was developed as part of the master thesis of [Nils Henkel](https://www.linkedin.com/in/nilshenkel/) at the [Institute of Structural Mechanics and Lightweight Design](https://www.sla.rwth-aachen.de/cms/~fald/Institut-fuer-Strukturmechanik-und-Leichtbau/?lidx=1) at RWTH Aachen University. 

# Requirements
- [Mutation++](https://github.com/mutationpp/Mutationpp) if you wish to construct own ablative materials (currently only available for Linux and macOS)
- Python 3 and some Python packages (mainly numpy, scipy, dill)

# Installation
If you wish to use ablative materials and "create" materials on your own, install [Mutation++](https://github.com/mutationpp/Mutationpp) first. This is necessary for pyrolysis gas enthalpy data as well as surface chemistry data.

For installation of Hot-STARSHIP open a terminal and navigate into the directory where you want to install Hot-STARSHIP using `cd`. Clone the repository using:
```
git clone https://github.com/nilsh7/Hot-STARSHIP.git
```
Install the required packages (possibly in your [virtual environment](https://docs.python.org/3/tutorial/venv.html)):
```
pip3 install -r requirements.txt
```

Next, set the environment variable `HOTSTARSHIP_DIR` to the directory where you installed Hot-STARSHIP:
```
export HOTSTARSHIP_DIR=/Users/<user>/<your_path_to_hot_starship>
```
In order to be able to call the program from any location, add the installation directory to your PYTHONPATH, too:
```
export PYTHONPATH=/Users/<user>/<your_path_to_hot_starship>:$PYTHONPATH
```

Per default, the output uses a custom matplotlib style for plots. [It is designed to be colorblind-friendly](https://personal.sron.nl/~pault/) and can be read by monochrome people or in monochrome printouts. To use the style copy the `Templates/MA_Style.mplstyle` file to the according style directory of matplotlib. For more information on the location of this directory check the [matplotlib documentation](https://matplotlib.org/tutorials/introductory/customizing.html#the-matplotlibrc-file).

# Usage
Using the program consists of three steps:
1. Preparing your case
2. Running your case
3. Viewing the results of your case

These are described in the following

## How to input your data
### Material properties
Material properties are specified by a set of ``.csv`` files that contain information on conductivity, specific heat capacity, emissivity and density.

#### Non-ablative
For materials that do not undergo pyrolysis or ablate, the material properties are set by a directory structure like the one below.
```bash
Aluminium
├── Aluminium.matp
├── cp
│   └── cp.csv
├── eps
│   └── eps.csv
├── k
│   └── k.csv
└── rho
    └── rho.csv
```
The ``.csv`` files shall contain the temperature in Kelvin in the first column and the respective property in SI units in the second column. The first row is skipped to include a header.

Please not that the thermal expansion is not considered. It is therefore advisable to use a non-temperature dependent behaviour.

Running ``material.py`` or ``hotstarship.py`` with the material as input will generate a `.matp` file in the directory of the material. This file is a ``dill``ed binary version of the ``Material`` object and contains all the information provided by the `.csv` files. The file can then be specified as input which significantly speeds up the process, especially for ablative materials.

#### Ablative
For ablative materials, the material properties of the virgin and char state might be different which is why there are two separate directories for these materials. In addition, information about the composition of the material and its decomposition reaction needs to be specified.
```bash
PICA
├── Char
│   ├── cp
│   │   └── cp.csv
│   ├── eps
│   │   └── eps.csv
│   └── k
│       └── k.csv
├── Combined
│   ├── Decomposition_Kinetics.csv
│   └── Heats_of_Formation.csv
├── Virgin
│   ├── comp
│   │   └── comp.csv
│   ├── cp
│   │   └── cp.csv
│   ├── eps
│   │   └── eps.csv
│   └── k
│       └── k.csv
├── PICA.matp
├── mutationpp_atmo_PICA.xml
├── mutationpp_gas_PICA.xml
└── mutationpp_surf_PICA.xml
```
The format of the conductivity, specific heat capacity and emissivity files is the same as the non-ablative case.

What you need to include in addition is:
1. in ``comp`` the composition of the material in mole fractions of C, H and O as well as the expected char yield
2. the heats of formation of virgin and char material and gas and the temperature at which these are given in ``Heats_of_Formation.csv``
3. properties related to the decomposition that is modelled by an Arrhenius law in ``Decomposition_Kinetics.csv`` 

The variables in the csv file are those of the following Arrhenius law equation:
<!--<img src="https://render.githubusercontent.com/render/math?math=\frac{\operatorname{d}&space;\rho_i}{\operatorname{d}&space;t}&space;=&space;-c_i&space;\left(&space;\frac{\rho_i&space;-&space;\rho_{c,i}}{\rho_{v,i}&space;-&space;\rho_{c,i}}&space;\right)^{n_{r,i}}&space;\mathrm{e}^{-\frac{B_i}{T}}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\frac{\operatorname{d}&space;\rho_i}{\operatorname{d}&space;t}&space;=&space;-c_i&space;\left(&space;\frac{\rho_i&space;-&space;\rho_{c,i}}{\rho_{v,i}&space;-&space;\rho_{c,i}}&space;\right)^{n_{r,i}}&space;\mathrm{e}^{-\frac{B_i}{T}}" title="\frac{\operatorname{d} \rho_i}{\operatorname{d} t} = -c_i \left( \frac{\rho_i - \rho_{c,i}}{\rho_{v,i} - \rho_{c,i}} \right)^{n_{r,i}} \mathrm{e}^{-\frac{B_i}{T}}">-->

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\operatorname{d}&space;\rho_i}{\operatorname{d}&space;t}&space;=&space;-c_i&space;\left(&space;\frac{\rho_i&space;-&space;\rho_{c,i}}{\rho_{v,i}&space;-&space;\rho_{c,i}}&space;\right)^{n_{r,i}}&space;\mathrm{e}^{-\frac{B_i}{T}}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\frac{\operatorname{d}&space;\rho_i}{\operatorname{d}&space;t}&space;=&space;-c_i&space;\left(&space;\frac{\rho_i&space;-&space;\rho_{c,i}}{\rho_{v,i}&space;-&space;\rho_{c,i}}&space;\right)^{n_{r,i}}&space;\mathrm{e}^{-\frac{B_i}{T}}" title="\frac{\operatorname{d} \rho_i}{\operatorname{d} t} = -c_i \left( \frac{\rho_i - \rho_{c,i}}{\rho_{v,i} - \rho_{c,i}} \right)^{n_{r,i}} \mathrm{e}^{-\frac{B_i}{T}}" /></a>
</p>

Please note that the definition of the c constant differs in literature and might need to be re-computed.
The total density is the sum of the densities multiplied with their respective volume fractions:

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\rho&space;=&space;v_1&space;\rho_1&space;&plus;&space;...&space;&plus;&space;v_2&space;\rho_2" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\rho&space;=&space;v_1&space;\rho_1&space;&plus;&space;...&space;&plus;&space;v_2&space;\rho_2" title="\rho = v_1 \rho_1 + ... + v_2 \rho_2" /></a>
</p>

The ``.xml`` and ``.matp`` files are generated as part of running the program.

### Input file
You can find a few sample input files in the ``Input/`` directory. They are xml files.

#### Defining layers
The first thing to be defined are the different layer that the thermal protection system consists of. Specify the material and thickness of each layer and give it a number from 1 at the surface to higher numbers on the inside. The first layer will follow an exponential cell length distribution defined by the first cell length (which is actually halved) and the maximum growth factor. The program will then fit the cells such that the growth factor takes this value as maximum.

The other layers use a constant growth factor, too, combined with the last cell thickness of the previous layer. The cells are fitted in the layer to achieve as little volume change as possible.

You can define ablative layers by using the ``ablative`` element. You may only define the first layer to be ablative in the current implementation.

Corrugated layers can be defined using the ``corrugated`` element. These layers than use a rule of mixtures to homogenize the material properties. You will need to specify the core as well as the web materials and some dimenions (see the `Input/input_corrugated.xml` file for this).


#### Boundary conditions
The first element in the option section are boundary conditions. For the front BC you may choose from ``heatflux``, `aerodynamic` and `recovery_enthalpy`.

The ``heatflux`` BC will simply create a flux at the surface. For ablative materials a blowing correction is used for which you will need to specify the transfer coefficient <a href="https://www.codecogs.com/eqnedit.php?latex=\rho_e&space;u_e&space;C_{H0}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\rho_e&space;u_e&space;C_{H0}" title="\rho_e u_e C_{H0}" /></a>. 

The `aerodynamic` BC calculates a constant transfer coefficient based on some temperatures that you will have to specify and the time at which these occur. This is only really achievable with results from CFD calculations.

The `recovery_enthalpy` BC specifies the recovery enthalpy of the boundary layer gases. It was only implemented for comparison with other codes. This might also be useful for validation with arc jet tests.

Note that all values can be specified as a numerical value or as a ``csv`` file that contains times in the first column and values in the second column in SI units. The first row is omitted when reading. 

#### Time stepping
Specifying the time steps can be achieved in two ways:
1. via ``start``, `end`, `delta` and `write_every` elements (the last one specifies how often the results shall be written)
2. via ``file`` element that specifies a `csv` file in which each time step is listed (first row is omitted as always)

#### Initilization
Initilization is currently only possible with a constant temperature that is specified using the `init` element.


#### Ambient conditions
The conditions of the environment are defined under the `ambient` element. This includes temperature that is part of the radiative exchange and pressure and atmosphere which the B' tables are generated with (only applicable for ablative materials; varying pressure not implemented yet). The `turbulent_flow` element is a boolean that affects the blowing correction.

## How to run Hot-STARSHIP
To run Hot-STARSHIP open a terminal window and type:
```
python3 hotstarship.py <input_file>.xml <output_file>.csv
```
You can use the ``-f`` option to force to overwrite the ouput file should it already exist. You will see some information about the number of necessary iterations in the console.

## Hot to view results
Running Hot-STARSHIP generates an output ``csv`` file that can be opened and read. The first column indicates the time. For each time step all nodal locations as well as the value of the variables are given. The back face that is identical with the last node is included, too. After the output of all nodal values, the time is incremented.

For ease of displaying results a number of functions is implemented in `output.py`. If you wish to use this libarary, type
```
python3
```
and
```
from hotstarship import output
```
Read your solution file using:
```
sr = output.SolutionReader(<path_to_your_solution_file>)
```
The ``SolutionReader`` objects provides plot tools that can be called using for instance
```
sr.plot('t', 'T', z=['Wall', 0.02, 0.05])
```
which will plot temperature as a function of time at the wall, 2 cm into the material and 5 cm into the material. If you rather wish to plot as function of location, go ahead:
```
sr.plot('z', 'beta', t=[20, 100, 200])
```
This plots the degree of char through the material after 20, 100 and 200 seconds.

Dependent variables include:
- `T`: temperature
- `rho`: density
- ``beta``: degree of char
- ``s``: recession amount
- ``sdot``: recession rate
- ```mc```: char mass flux
- ```mg```: pyrolysis gas mass flux

In addition the following functions of the ``SolutionReader`` object are available:
1. ``calculate_mass(t)``: returns the mass per unit area at time `t
2.  ``get_max_back_T()``: returns the maximum back face temperature
3. ```get_max_T()```: returns the global maximum temperature
4. `get_remaining_thickness(t)`: returns the remaining TPS thickness at time `t` 

# Sources
- Amar, Adam Joseph. "Modeling of one-dimensional ablation with porous flow using finite control volume procedure." (2006).
- Chen, Y-K., and Frank S. Milos. "Ablation and thermal response program for spacecraft heatshield analysis." Journal of Spacecraft and Rockets 36.3 (1999): 475-483.

# Disclaimer
The author does not warrant the correctness of the results, nor is he liable for any damage using this program's results. This program was made to the author's best knowledge and belief. 