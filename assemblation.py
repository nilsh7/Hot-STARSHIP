import material
import numpy as np
from scipy.sparse import dia_matrix, lil_matrix

p1 = lambda f: np.roll(f, shift=-1)
m1 = lambda f: np.roll(f, shift=+1)
p12 = lambda f: (p1(f) + f)/2
m12 = lambda f: (m1(f) + f)/2

def addVariables(layers):

    for layer in layers:
        layer.wv = np.ones(layer.grid.nC)

    return layers

def createUnknownVectors(layers):

    # For Tmap
    Tmap = {}

    counter = 0

    if layers[0].ablative:
        Tmap["sdot"] = counter
        counter += 1

    for layer in layers:
        Tmap["lay" + str(layer.number)] = np.arange(counter, counter+layer.grid.nC)
        if layer is not layers[-1]:
            Tmap["int" + str(layer.number)] = counter+layer.grid.nC
        counter += layer.grid.nC + 1

    numTs = Tmap["lay" + str(layers[-1].number)][-1] + 1

    # For rhomap
    rhomap = {}

    counter = 0

    for layer in layers:
        rhomap["lay" + str(layer.number)] = np.arange(counter, counter + layer.grid.nC)
        counter += layer.grid.nC

    numRhos = rhomap["lay" + str(layers[-1].number)][-1] + 1

    T, rho = np.zeros(numTs), np.zeros(numRhos)

    return T, rho, Tmap, rhomap


def init_T_rho(T, rho, Tmap, rhomap, layers, inputvars):
    if inputvars.initType == "Temperature":
        if "sdot" in Tmap:
            T[0] = 0
            T[1:] = inputvars.initValue
        else:
            T[:] = inputvars.initValue
    else:
        raise ValueError("Unimplemented initialization type %s", inputvars.initType)
    for layerKey, layer in zip(rhomap, layers):
        if type(layer.material) is material.AblativeMaterial:
            rho[rhomap[layerKey]] = np.array([np.dot(layer.material.data.virginRho0, layer.material.data.frac)]*len(rhomap[layerKey]))
        else:
            if inputvars.initType == "Temperature":
                rho[rhomap[layerKey]] = np.array([layer.material.rho(inputvars.initValue)]*len(rhomap[layerKey]))
            else:
                raise ValueError("Unimplemented initialization type %s", inputvars.initType)

    return T, rho

def assembleT(layers, Tmap, Tnu, Tn, tDelta, inputvars):

    numUnknowns = []

    J = assembleTMatrix(layers, Tmap, Tnu, tDelta, inputvars)

    fnu = assembleTVector()

    return J, fnu


def assembleTMatrix(layers, Tmap, Tnu, tDelta, inputvars):

    J = dia_matrix((len(Tnu), len(Tnu)))
    first_col = np.zeros(len(Tnu))
    J_int = lil_matrix((len(Tnu), len(Tnu)))

    for key in Tmap:



        if key == "sdot":
            pass
        elif key[0:3] == "lay":

            # Store some variables
            iStart = Tmap[key][0]
            iEnd = Tmap[key][-1]
            lay = layers[int(key[3:])]
            mat = lay.material
            gr = lay.grid
            Tj = Tnu[Tmap[key]]

            ### Conduction ###
            # Flux at plus side
            dCjp12_dTj = mat.k(p12(Tj), p12(lay.wv))/gr.dzjp + (Tj - p1(Tj)) / (2*gr.dzjp) * mat.dkdT(Tj, lay.wv)
            dCjp12_dTjp1 = -mat.k(p12(Tj), p12(lay.wv))/gr.dzjp + (Tj - p1(Tj)) / (2*gr.dzjp) * mat.dkdT(p1(Tj), p1(lay.wv))
            if lay.ablative:
                dCjp12_dsdot = -mat.k(p12(Tj), p12(lay.wv)) * (Tj - p1(Tj)) / (gr.dzjp**2) * (p1(gr.etaj) - gr.etaj) * tDelta
            else:
                dCjp12_dsdot = np.zeros(len(Tj))
            dCjp12_dTj[-1], dCjp12_dTjp1[-1], dCjp12_dsdot[-1] = (0, 0, 0)

            # Flux at minus side
            dCjm12_dTj = -mat.k(m12(Tj), m12(lay.wv))/gr.dzjm + (m1(Tj) - Tj) / (2*gr.dzjm) * mat.dkdT(Tj, lay.wv)
            dCjm12_dTjm1 = mat.k(m12(Tj), m12(lay.wv)) / gr.dzjm + (m1(Tj) - Tj) / (2 * gr.dzjm) * mat.dkdT(m1(Tj), m1(lay.wv))
            if lay.ablative:
                dCjm12_dsdot = -mat.k(m12(Tj), m12(lay.wv)) * (m1(Tj) - Tj) / (gr.dzjm**2) * (gr.etaj - m1(gr.etaj)) * tDelta
            else:
                dCjm12_dsdot = np.zeros(len(Tj))
            dCjm12_dTj[0], dCjm12_dTjm1[-1], dCjm12_dsdot[-1] = (0, 0, 0)

            # Assemble Jacobian matrix
            fluxes = (dCjp12_dTj, dCjp12_dTjp1, dCjm12_dTj, dCjm12_dTjm1)
            signs = (+1, +1, -1, -1)
            offsets = (0, +1, 0, -1)
            for flux, sign, offset in zip(fluxes, signs, offsets):
                globflux = np.zeros(len(Tnu))
                globflux[iStart:iEnd+1] = flux
                J += dia_matrix((sign * globflux, offset), shape=(len(Tnu), len(Tnu)))

            # Store first column in separate vector
            first_col[iStart:iEnd+1] += dCjp12_dsdot + dCjm12_dsdot

            aaa = 1

        elif key[0:3] == "int":

            # Store some variables
            iInt = Tmap[key]
            lay_left = layers[int(key[3:])]
            lay_right = layers[int(key[3:])+1]
            T_left = Tnu[Tmap["lay" + key[3:]]][-1]
            T_right = Tnu[Tmap["lay" + str(int(key[3:])+1)]][0]
            T_int = Tnu[Tmap[key]]
            wv = lay_left.wv[-1]

            # Left side conduction flux
            dz_left = (lay_left.grid.zjp12[-1] - lay_left.grid.zj[-1])
            dCl_dTl = lay_left.material.k(T_int, wv) / dz_left
            dCl_dTint = -lay_left.material.k(T_int, wv) / dz_left + lay_left.material.dkdT(T_int, wv) * \
                        (T_left - T_int) / dz_left
            dCl_dsdot = lay_left.grid.etaj[-1] * tDelta if lay_left.ablative else 0

            # Right side conduction flux
            dz_right = lay_right.grid.zj[0] - lay_right.grid.zjm12[0]
            dCr_dTr = -lay_right.material.k(T_int, 1.0) / dz_right
            dCr_dTint = +lay_right.material.k(T_int, 1.0) / dz_right + lay_right.material.dkdT(T_int, 1.0) * \
                        (T_int - T_right) / dz_right
            dCr_dsdot = 0

            # Assemble Jacobian matrix
            J_int[iInt-1, iInt-1] += +dCl_dTl  # Contribution to energy balance of previous volume
            J_int[iInt-1, iInt]   += +dCl_dTint  # Contribution to energy balance of previous volume
            J_int[iInt,   iInt-1] += -dCl_dTl  # Contribution to energy balance of interface
            J_int[iInt,   iInt]   += -dCl_dTint  # Contribution to energy balance of interface
            J_int[iInt,   iInt]   += +dCr_dTint  # Contribution to energy balance of interface
            J_int[iInt,   iInt+1] += +dCr_dTr  # Contribution to energy balance of interface
            J_int[iInt+1, iInt]   += +dCr_dTint  # Contribution to energy balance of next volume
            J_int[iInt+1, iInt+1] += +dCr_dTr  # Contribution to energy balance of next volume

            first_col[iInt-1] += +dCl_dsdot
            first_col[iInt] += -dCl_dsdot + dCr_dsdot
            first_col[iInt+1] += -dCr_dsdot


def assembleTVector():
    pass
