import material
import numpy as np

def createUnknownVectors(layers, ablative):

    # For Tmap
    Tmap = {}

    counter = 0

    if ablative:
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

def assembleT(inputvars):

    numUnknowns = []

    J = assembleTMatrix()

    fmu = assembleTVector()

    return J, fmu


def assembleTMatrix():
    pass


def assembleTVector():
    pass
