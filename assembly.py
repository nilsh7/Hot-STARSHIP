import material
import numpy as np
from scipy.sparse import dia_matrix, lil_matrix
import grid

p1 = lambda f: np.roll(f, shift=-1)
m1 = lambda f: np.roll(f, shift=+1)
p12 = lambda f: (p1(f) + f)/2
m12 = lambda f: (m1(f) + f)/2
sw = 1  # stencil width

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
        Tmap["lay%i" % layer.number] = np.arange(counter, counter+layer.grid.nC)
        if layer is not layers[-1]:
            Tmap["int%i" % layer.number] = counter+layer.grid.nC
        counter += layer.grid.nC + 1

    numTs = Tmap["lay%i" % layers[-1].number][-1] + 1

    # For rhomap
    rhomap = {}

    counter = 0

    for layer in layers:
        rhomap["lay%i" % layer.number] = np.arange(counter, counter + layer.grid.nC)
        counter += layer.grid.nC

    numRhos = rhomap["lay%i" % layers[-1].number][-1] + 1

    T, rho = np.zeros(numTs), np.zeros(numRhos)

    return T, rho, Tmap, rhomap


def createGlobFlux(flux, length, iStart, iEnd, offset):

    globflux = np.zeros(length + sw)
    if iStart+offset < 0:
        st = 0
        trunc = -(iStart+offset)
    else:
        st = iStart + offset
        trunc = 0

    globflux[st:iEnd + offset + 1] = flux[trunc:]

    return globflux


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
            rho[rhomap[layerKey]] = np.repeat(np.dot(layer.material.data.virginRhoFrac0, layer.material.data.frac),
                                              len(rhomap[layerKey]))
            rhoi = np.repeat(layer.material.data.virginRhoFrac0.reshape(1, -1), repeats=len(rhomap[layerKey]), axis=0)
            mgas = np.zeros(len(rhomap[layerKey]))
        else:
            if inputvars.initType == "Temperature":
                rho[rhomap[layerKey]] = np.repeat(layer.material.rho(inputvars.initValue), len(rhomap[layerKey]))
                rhoi = np.zeros(len(rhomap[layerKey]))
            else:
                raise ValueError("Unimplemented initialization type %s", inputvars.initType)

    return T, rho, rhoi, mgas


def assembleT(layers, layerspre, Tmap, Tnu, Tn, rhomap, rhonu, rhon, mgas, t, tDelta, inputvars):

    J = assembleTMatrix(layers, Tmap, Tnu, rhonu, rhomap, mgas, tDelta, inputvars)

    fnu = assembleTVector(layers, layerspre, Tnu, Tn, Tmap, rhonu, rhon, rhomap, mgas, t, tDelta, inputvars)

    return J, fnu


def assembleTMatrix(layers, Tmap, Tnu, rhonu, rhomap, mgas, tDelta, inputvars):

    J = dia_matrix((len(Tnu), len(Tnu)))
    first_col = np.zeros(len(Tnu))

    for key in Tmap:

        if key == "sdot":

            if inputvars.BCfrontType == "aerodynamic":

                J, first_col = addWallBlowMatrix(J, first_col, Tnu, Tmap, rhonu, rhomap, mgas, layers, inputvars)

        elif key[0:3] == "lay":

            J, first_col = addConductionMatrixInner(J, first_col, Tnu, Tmap, layers, key, tDelta)

            J, first_col = addEnergyMatrix(J, first_col, Tnu, Tmap, rhonu, rhomap, layers, key, tDelta, inputvars)

            J, first_col = addGridMatrix(J, first_col, Tnu, Tmap, rhonu, rhomap, layers, key)

            J = addPyroMatrix(J, Tnu, Tmap, mgas, layers, key, inputvars)

        elif key[0:3] == "int":

            J, first_col = addConductionMatrixOuter(J, first_col, Tnu, Tmap, layers, key, tDelta)

    # Add BC
    if inputvars.BCfrontType == "aerodynamic":

        J, first_col = addAerodynamicMatrix(J, first_col, Tnu, Tmap, rhonu, rhomap, mgas, layers, inputvars)

        J, first_col = addBcMatrix(J, first_col, Tnu, Tmap, rhonu, rhomap, mgas, layers, inputvars)

    elif inputvars.BCfrontType == "heatflux":
        pass  # No sensitivity of heat flux
    else:
        raise ValueError("Unimplemented front BC %s" % inputvars.BCfrontType)

    Jlil = J.tolil()
    Jlil[:, 0] += first_col.reshape(-1, 1)
    Jcsc = Jlil.tocsc()

    return Jcsc


def assembleTVector(layers, layerspre, Tnu, Tn, Tmap, rhonu, rhon, rhomap, mgas, t, tDelta, inputvars):

    fnu = np.zeros(len(Tnu))

    for key in Tmap:

        if key == "sdot":

            if inputvars.BCfrontType == "aerodynamic":
                fnu = addWallBlowVector(fnu, Tnu, Tmap, rhonu, rhomap, mgas, layers, inputvars)

        elif key[0:3] == "lay":

            fnu = addConductionVectorInner(fnu, Tnu, Tmap, layers, key)

            fnu = addEnergyVector(fnu, Tnu, Tn, Tmap, rhonu, rhon, rhomap, layers, layerspre, key, tDelta, inputvars)

            fnu = addGridVector(fnu, Tnu, Tmap, rhonu, rhomap, layers, key)

            fnu = addPyroVector(fnu, Tnu, Tmap, mgas, layers, key, inputvars)

        elif key[0:3] == "int":

            fnu = addConductionVectorOuter(fnu, Tnu, Tmap, layers, key)

    # Add BC
    if inputvars.BCfrontType == "aerodynamic":

        fnu = addAerodynamicVector(fnu, Tnu, Tmap, rhonu, rhomap, mgas, layers, inputvars)

        fnu = addBcVector(fnu, Tnu, Tmap, rhonu, rhomap, mgas, layers, inputvars)

    elif inputvars.BCfrontType == "heatflux":
        fnu = addHeatFluxVector(fnu, Tmap, inputvars, t)
    else:
        raise ValueError("Unimplemented front BC %s" % inputvars.BCfrontType)

    return fnu


def addConductionMatrixInner(J, first_col, Tnu, Tmap, layers, key, tDelta):

    # Store some variables
    iStart = Tmap[key][0]
    iEnd = Tmap[key][-1]
    lay = layers[int(key[3:])]
    mat = lay.material
    gr = lay.grid
    Tj = Tnu[Tmap[key]]

    ### Conduction ###
    # Flux at plus side
    dCjp12_dTj   =  mat.k(p12(Tj), p12(lay.wv)) / gr.dzjp + (Tj - p1(Tj)) / (2 * gr.dzjp) * mat.dkdT(Tj, lay.wv)  # TODO: change to p12(lay.wv)
    dCjp12_dTjp1 = -mat.k(p12(Tj), p12(lay.wv)) / gr.dzjp + (Tj - p1(Tj)) / (2 * gr.dzjp) * mat.dkdT(p1(Tj), p1(lay.wv))
    if lay.ablative:
        dCjp12_dsdot = -mat.k(p12(Tj), p12(lay.wv)) * (Tj - p1(Tj)) / (gr.dzjp ** 2) * (p1(gr.etaj) - gr.etaj) * tDelta
    else:
        dCjp12_dsdot = np.zeros(len(Tj))
    dCjp12_dTj[-1], dCjp12_dTjp1[-1], dCjp12_dsdot[-1] = (0, 0, 0)

    # Flux at minus side
    dCjm12_dTj   = -mat.k(m12(Tj), m12(lay.wv)) / gr.dzjm + (m1(Tj) - Tj) / (2 * gr.dzjm) * mat.dkdT(Tj, lay.wv)
    dCjm12_dTjm1 = +mat.k(m12(Tj), m12(lay.wv)) / gr.dzjm + (m1(Tj) - Tj) / (2 * gr.dzjm) * mat.dkdT(m1(Tj), m1(lay.wv))
    if lay.ablative:
        dCjm12_dsdot = -mat.k(m12(Tj), m12(lay.wv)) * (m1(Tj) - Tj) / (gr.dzjm ** 2) * (gr.etaj - m1(gr.etaj)) * tDelta
    else:
        dCjm12_dsdot = np.zeros(len(Tj))
    dCjm12_dTj[0], dCjm12_dTjm1[0], dCjm12_dsdot[0] = (0, 0, 0)

    # Assemble Jacobian matrix
    fluxes = (dCjp12_dTj, dCjp12_dTjp1, dCjm12_dTj, dCjm12_dTjm1)
    signs = (+1, +1, -1, -1)
    offsets = (0, +1, 0, -1)
    for flux, sign, offset in zip(fluxes, signs, offsets):
        globflux = createGlobFlux(flux, len(Tnu), iStart, iEnd, offset)
        J += dia_matrix((sign * globflux, offset), shape=(len(Tnu), len(Tnu)))

    # Store first column in separate vector
    first_col[iStart:iEnd + 1] += dCjp12_dsdot - dCjm12_dsdot

    return J, first_col


def addConductionMatrixOuter(J, first_col, Tnu, Tmap, layers, key, tDelta):

    J_int = lil_matrix((len(Tnu), len(Tnu)))

    # Store some variables
    iInt = Tmap[key]
    lay_left = layers[int(key[3:])]
    lay_right = layers[int(key[3:]) + 1]
    T_left = Tnu[Tmap["lay%s" % key[3:]]][-1]
    T_right = Tnu[Tmap["lay%i" % (int(key[3:]) + 1)]][0]
    T_int = Tnu[Tmap[key]]
    wv = lay_left.wv[-1]

    # Left side conduction flux
    dz_left = (lay_left.grid.zjp12[-1] - lay_left.grid.zj[-1])
    dCl_dTl = lay_left.material.k(T_int, wv) / dz_left
    dCl_dTint = -lay_left.material.k(T_int, wv) / dz_left + lay_left.material.dkdT(T_int, wv) * \
                (T_left - T_int) / dz_left
    dCl_dsdot = -lay_left.material.k(T_int, wv) / dz_left**2 * (T_left - T_int) * \
               (-lay_left.grid.etaj[-1] * tDelta) if lay_left.ablative else 0

    # Right side conduction flux
    dz_right = lay_right.grid.zj[0] - lay_right.grid.zjm12[0]
    dCr_dTr = -lay_right.material.k(T_int, 1.0) / dz_right
    dCr_dTint = +lay_right.material.k(T_int, 1.0) / dz_right + lay_right.material.dkdT(T_int, 1.0) * \
                (T_int - T_right) / dz_right
    dCr_dsdot = 0

    # Assemble Jacobian matrix
    J_int[iInt - 1, iInt - 1] += +dCl_dTl  # Contribution to energy balance of previous volume
    J_int[iInt - 1, iInt] += +dCl_dTint  # Contribution to energy balance of previous volume
    J_int[iInt, iInt - 1] += -dCl_dTl  # Contribution to energy balance of interface
    J_int[iInt, iInt] += -dCl_dTint  # Contribution to energy balance of interface
    J_int[iInt, iInt] += +dCr_dTint  # Contribution to energy balance of interface
    J_int[iInt, iInt + 1] += +dCr_dTr  # Contribution to energy balance of interface
    J_int[iInt + 1, iInt] += +dCr_dTint  # Contribution to energy balance of next volume
    J_int[iInt + 1, iInt + 1] += +dCr_dTr  # Contribution to energy balance of next volume

    first_col[iInt - 1] += +dCl_dsdot
    first_col[iInt] += -dCl_dsdot + dCr_dsdot
    first_col[iInt + 1] += -dCr_dsdot

    J += J_int.todia()

    return J, first_col


def addConductionVectorInner(fnu, Tnu, Tmap, layers, key):

    # Store some variables
    iStart = Tmap[key][0]
    iEnd = Tmap[key][-1]
    lay = layers[int(key[3:])]
    mat = lay.material
    gr = lay.grid
    Tj = Tnu[Tmap[key]]

    ### Conduction ###
    # Flux at plus side
    Cjp12 = mat.k(p12(Tj), p12(lay.wv)) / gr.dzjp * (Tj - p1(Tj))
    # Flux at minus side
    Cjm12 = mat.k(m12(Tj), m12(lay.wv)) / gr.dzjm * (m1(Tj) - Tj)
    Cjp12[-1], Cjm12[0] = (0, 0)  # Account for boundaries, will be added in addConductionVectorOuter

    # Assemble vector
    fnu[iStart:iEnd+1] += -Cjm12 + Cjp12

    return fnu


def addConductionVectorOuter(fnu, Tnu, Tmap, layers, key):

    # Store some variables
    iInt = Tmap[key]
    lay_left = layers[int(key[3:])]
    lay_right = layers[int(key[3:]) + 1]
    T_left = Tnu[Tmap["lay%s" % key[3:]]][-1]
    T_right = Tnu[Tmap["lay%i" % (int(key[3:]) + 1)]][0]
    T_int = Tnu[Tmap[key]]
    wv = lay_left.wv[-1]

    ### Conduction ###
    # Left side conduction flux
    dz_left = (lay_left.grid.zjp12[-1] - lay_left.grid.zj[-1])
    Cl = lay_left.material.k(T_int, wv) / dz_left * (T_left - T_int)

    # Right side conduction flux
    dz_right = lay_right.grid.zj[0] - lay_right.grid.zjm12[0]
    Cr = lay_right.material.k(T_int, wv) / dz_right * (T_int - T_right)

    # Assemble vector
    fnu[iInt-1] += +Cl  # Energy balance of previous volume
    fnu[iInt]   += -Cl + Cr  # Energy balance of interface
    fnu[iInt+1] += -Cr  # Energy balance of next volume

    return fnu


def addEnergyMatrix(J, first_col, Tnu, Tmap, rhonu, rhomap, layers, key, tDelta, inputvars):

    # Store some variables
    iStart = Tmap[key][0]
    iEnd = Tmap[key][-1]
    lay = layers[int(key[3:])]
    mat = lay.material
    gr = lay.grid
    Tj = Tnu[Tmap[key]]
    rhoj = rhonu[rhomap[key]]

    # Derivatives
    if lay.ablative:
        dEj_dsdot = ((gr.etaj - m1(gr.etaj)) * (3/8 * rhoj * mat.e(Tj, lay.wv) + 1/8 * m1(rhoj) * m1(mat.e(Tj, lay.wv))) +
                     (p1(gr.etaj) - gr.etaj) * (3/8 * rhoj * mat.e(Tj, lay.wv) + 1/8 * p1(rhoj) * p1(mat.e(Tj, lay.wv))))
    else:
        dEj_dsdot = np.zeros(len(rhoj))

    dEj_dTj = (gr.dzjm + gr.dzjp) * 3/8 * rhoj * mat.cp(Tj, lay.wv) / tDelta
    dEj_dTjm1 = gr.dzjm * 1/8 * m1(rhoj) * m1(mat.cp(Tj, lay.wv)) / tDelta
    dEj_dTjp1 = gr.dzjp * 1/8 * p1(rhoj) * p1(mat.cp(Tj, lay.wv)) / tDelta
    # Note: values at index 0 and -1 have no physical meaning!

    ### Calculate fluxes at boundaries ###
    # Minus side
    if type(gr) == grid.FrontGrid:
        # Interface at minus side
        # Half cell has nothing in negative direction
        if lay.ablative:
            dEj_dsdot[0] = (gr.etaj[1]-gr.etaj[0]) * (3/8 * rhoj[0] * mat.e(Tj[0], lay.wv[0]) +
                                                               1/8 * rhoj[1] * mat.e(Tj[1], lay.wv[1]))
        dEj_dTj[0] = gr.dzjp[0] * 3/8 * rhoj[0] * mat.cp(Tj[0], lay.wv[0]) / tDelta
        dEj_dTjm1[0] = 0
        dEj_dTjp1[0] = gr.dzjp[0] * 1/8 * rhoj[1] * mat.cp(Tj[1], lay.wv[1]) / tDelta
    elif type(gr) == grid.DeepGrid and int(key[3:]) != 0:

        # Get previous interface
        intkey = "int%i" % (int(key[3:])-1)

        # Interface at minus side
        # Assign some values
        Tint = Tnu[Tmap[intkey]]
        dzint = gr.zj[0] - gr.zjm12[0]
        dzp = gr.dzjp[0]

        dEj_dTj[0] = rhoj[0] * mat.cp(Tj[0], lay.wv[0]) * (1/2 * dzint + 3/8 * dzp) / tDelta
        dEj_dTjm1[0] = rhoj[0] * mat.cp(Tint, lay.wv[0]) * 1/2 * dzint / tDelta
        dEj_dTjp1[0] = rhoj[1] * mat.cp(Tj[1], lay.wv[1]) * 1/8 * dzp / tDelta
        if lay.ablative:
            raise UserWarning("Deeper layers cannot be ablative.")
    else:
        raise UserWarning("DeepGrid at front is not supported yet.")

    # Plus side
    intkey = "int%s" % key[3:]
    if intkey in Tmap:
        # Boundary in middle of material
        # Interface at plus side
        # Assign some values
        Tint = Tnu[Tmap[intkey]]
        dzint = gr.zjp12[-1] - gr.zj[-1]
        dzm = gr.dzjm[-1]

        # Calculate derivatives
        dEj_dTj[-1] = rhoj[-1] * mat.cp(Tj[-1], lay.wv[-1]) * (dzint/2 + 3/8 * dzm * 3/8) / tDelta
        dEj_dTjm1[-1] = dzm * 1/8 * rhoj[-2] * mat.cp(Tj[-2], lay.wv[-2]) / tDelta
        dEj_dTjp1[-1] = rhoj[-1] * mat.cp(Tint, lay.wv[-1]) * dzint/2 / tDelta
        if lay.ablative:
            dEj_dsdot[-1] = ((rhoj[-1] * (mat.e(Tint, lay.wv[-1]) + mat.e(Tj[-1], lay.wv[-1]))/2) * (-gr.etaj[-1]) +
                             (gr.etaj[-1] - gr.etaj[-2]) * (3/8 * rhoj[-1] * mat.e(Tj[-1], lay.wv[-1]) +
                                                            1/8 * rhoj[-2] * mat.e(Tj[-2], lay.wv[-2])))
    else:
        # Boundary at back of material
        if inputvars.BCbackType == "adiabatic":
            dzp12 = gr.zjp12[-1] - gr.zj[-1]
            dzm = gr.dzjm[-1]

            dEj_dTj[-1] = (dzm * 3/8 + dzp12) * rhoj[-1] * mat.cp(Tj[-1], lay.wv[-1]) / tDelta
            dEj_dTjm1[-1] = dzm * 1/8 * rhoj[-2] * mat.cp(Tj[-2], lay.wv[-2]) / tDelta
            dEj_dTjp1[-1] = 0

            if lay.ablative:
                dEj_dsdot[-1] = ((gr.etaj[-1] - gr.etaj[-2]) * (3/8 * rhoj[-1] * mat.e(Tj[-1], lay.wv[-1]) +
                                                                1/8 * rhoj[-2] * mat.e(Tj[-2], lay.wv[-2])) +
                                 (0 - gr.etaj[-1]) * rhoj[-1] * mat.e(Tj[-1], lay.wv[-1]))
            else:
                dEj_dsdot[-1] = 0
        else:
            raise UserWarning("Back BC %s not implemented yet." % inputvars.BCbackType)

    # Assemble Jacobian matrix
    fluxes = (dEj_dTj, dEj_dTjm1, dEj_dTjp1)
    signs = (+1, +1, +1)
    offsets = (0, -1, +1)
    for flux, sign, offset in zip(fluxes, signs, offsets):
        globflux = createGlobFlux(flux, len(Tnu), iStart, iEnd, offset)
        J += dia_matrix((sign * globflux, offset), shape=(len(Tnu), len(Tnu)))

    # Store first column in separate vector
    first_col[iStart:iEnd + 1] += dEj_dsdot

    return J, first_col


def addEnergyVector(fnu, Tnu, Tn, Tmap, rhonu, rhon, rhomap, layers, layerspre, key, tDelta, inputvars):

    # Store some variables
    iStart = Tmap[key][0]
    iEnd = Tmap[key][-1]
    lay = layers[int(key[3:])]
    laypre = layerspre[int(key[3:])]
    mat = lay.material
    gr = lay.grid
    grpre =  laypre.grid
    Tjnu = Tnu[Tmap[key]]
    Tjn = Tn[Tmap[key]]
    rhojnu = rhonu[rhomap[key]]
    rhojn = rhon[rhomap[key]]

    # Values
    Ej = 1/tDelta * (+gr.dzjm    * (+3/8 * rhojnu * mat.e(Tjnu, lay.wv)
                                    +1/8 * m1(rhojnu) * mat.e(m1(Tjnu), m1(lay.wv)))
                     -grpre.dzjm * (+3/8 * rhojn * mat.e(Tjn, laypre.wv)
                                    +1/8 * m1(rhojn) * mat.e(m1(Tjn), m1(laypre.wv)))
                     +gr.dzjp    * (+3/8 * rhojnu * mat.e(Tjnu, lay.wv)
                                    +1/8 * p1(rhojnu) * mat.e(p1(Tjnu), p1(lay.wv)))
                     -grpre.dzjp * (+3/8 * rhojn * mat.e(Tjn, laypre.wv)
                                    +1/8 * p1(rhojn) * mat.e(p1(Tjn), p1(laypre.wv))))
    # Note: values at index 0 and -1 have no physical meaning!

    ### Calculate fluxes at boundaries ###
    # Minus side
    if type(gr) == grid.FrontGrid:
        Ej[0] = 1/tDelta * (+gr.dzjp[0]    * (+3/8 * rhojnu[0] * mat.e(Tjnu[0], lay.wv[0])
                                              +1/8 * rhojnu[1] * mat.e(Tjnu[1], lay.wv[1]))
                            -grpre.dzjp[0] * (+3/8 * rhojn[0] * mat.e(Tjn[0], laypre.wv[0])
                                              +1/8 * rhojn[1] * mat.e(Tjn[1], laypre.wv[1])))
    elif type(gr) == grid.DeepGrid and int(key[3:]) != 0:

        # Get previous interface
        intkey = "int%i" % (int(key[3:]) - 1)

        # Interface at minus side
        # Assign some values
        Tintnu = Tnu[Tmap[intkey]]
        Tintn = Tn[Tmap[intkey]]
        dzintnu = gr.zj[0] - gr.zjm12[0]
        dzintn = grpre.zj[0] - grpre.zjm12[0]

        Ej[0] = 1/tDelta * (+ rhojnu[0] * (mat.e(Tintnu, lay.wv[0])+mat.e(Tjnu[0], lay.wv[0]))/2 * dzintnu
                            + (gr.dzjp[0] * (3/8 * rhojnu[0] * mat.e(Tjnu[0], lay.wv[0]) +
                                             1/8 * rhojnu[1] * mat.e(Tjnu[1], lay.wv[1])))
                            - rhon[0] * (mat.e(Tintn, laypre.wv[0])+mat.e(Tjn[0], laypre.wv[0]))/2 * dzintn
                            - (grpre.dzjp[0] * (3/8 * rhojn[0] * mat.e(Tjn[0], laypre.wv[0]) +
                                                1/8 * rhojn[1] * mat.e(Tjn[1], laypre.wv[1]))))
    else:
        raise UserWarning("DeepGrid at front is not supported yet.")

    # Plus side
    intkey = "int%s" % key[3:]
    if intkey in Tmap:
        # Boundary in middle of material
        # Interface at plus side
        # Assign some values
        Tintnu = Tnu[Tmap[intkey]]
        Tintn = Tn[Tmap[intkey]]
        dzintnu = gr.zjp12[-1] - gr.zj[-1]
        dzintn = grpre.zjp12[-1] - grpre.zj[-1]

        # Calculate derivatives
        Ej[-1] = 1/tDelta * (+ rhojnu[-1] * (mat.e(Tintnu, lay.wv[-1])+mat.e(Tjnu[-1], lay.wv[-1]))/2 * dzintnu
                             + (gr.dzjm[-1] * (3/8 * rhojnu[-1] * mat.e(Tjnu[-1], lay.wv[-1]) +
                                               1/8 * rhojnu[-2] * mat.e(Tjnu[-2], lay.wv[-2])))
                             - rhon[-1] * (mat.e(Tintn, laypre.wv[-1])+mat.e(Tjn[-1], laypre.wv[-1]))/2 * dzintn
                             - (grpre.dzjm[-1] * (3/8 * rhojn[-1] * mat.e(Tjn[-1], laypre.wv[-1]) +
                                                  1/8 * rhojn[-2] * mat.e(Tjn[-2], laypre.wv[-2]))))
    else:
        # Boundary at back of material
        if inputvars.BCbackType == "adiabatic":
            dzp12nu = gr.zjp12[-1] - gr.zj[-1]
            dzp12n = grpre.zjp12[-1] - grpre.zj[-1]

            Ej[-1] = 1/tDelta * (+ gr.dzjm[-1] * (3/8 * rhojnu[-1] * mat.e(Tjnu[-1], lay.wv[-1]) +
                                                  1/8 * rhojnu[-2] * mat.e(Tjnu[-2], lay.wv[-2]))
                                 + rhojnu[-1] * mat.e(Tjnu[-1], lay.wv[-1]) * 1/2 * dzp12nu
                                 - grpre.dzjm[-1] * (3/8 * rhojn[-1] * mat.e(Tjn[-1], laypre.wv[-1]) +
                                                     1/8 * rhojn[-2] * mat.e(Tjn[-2], laypre.wv[-2]))
                                 - rhojn[-1] * mat.e(Tjn[-1], lay.wv[-1]) * 1/2 * dzp12n)
        else:
            raise UserWarning("Back BC %s not implemented yet." % inputvars.BCbackType)

    # Assemble vector
    fnu[iStart:iEnd + 1] += +Ej

    return fnu


def addGridMatrix(J, first_col, Tnu, Tmap, rhonu, rhomap, layers, key):

    lay = layers[int(key[3:])]
    # The deeper layers are not moving, nor is the front layer if we don't have an ablative case
    # Thus we don't have any convection due to grid movement
    if int(key[3:]) != 0 and not lay.ablative:
        return J, first_col

    # Store some variables
    iStart = Tmap[key][0]
    iEnd = Tmap[key][-1]
    mat = lay.material
    gr = lay.grid
    sdot = Tnu[Tmap["sdot"]]
    Tj = Tnu[Tmap[key]]
    rhoj = rhonu[rhomap[key]]

    ### Grid movement ###
    # Flux at plus side
    dGjp12_dTj = 1/2 * rhoj * mat.cp(Tj, lay.wv) * gr.etajp12 * sdot
    dGjp12_dTjp1 = 1 / 2 * p1(rhoj) * p1(mat.cp(Tj, lay.wv)) * gr.etajp12 * sdot
    if lay.ablative:
        dGjp12_dsdot = 1/2 * (rhoj * mat.e(Tj, lay.wv) + p1(rhoj) * p1(mat.e(Tj, lay.wv))) * gr.etajp12
    else:
        dGjp12_dsdot = np.zeros(len(Tj))

    # Value at index -1 should naturally be zero due to gr.etajp12[-1] = 0

    # Flux at minus side
    dGjm12_dTj = 1 / 2 * rhoj * mat.cp(Tj, lay.wv) * gr.etajm12 * sdot
    dGjm12_dTjm1 = 1 / 2 * m1(rhoj) * m1(mat.cp(Tj, lay.wv)) * gr.etajm12 * sdot
    if lay.ablative:
        dGjm12_dsdot = 1 / 2 * (rhoj * mat.e(Tj, lay.wv) + m1(rhoj) * m1(mat.e(Tj, lay.wv))) * gr.etajm12
    else:
        dGjm12_dsdot = np.zeros(len(Tj))

    dGjm12_dTj[0] = rhoj[0] * mat.cp(Tj[0], lay.wv[0]) * sdot
    dGjm12_dTjm1[0] = 0
    if lay.ablative:
        dGjm12_dsdot[0] = rhoj[0] * mat.e(Tj[0], lay.wv[0])

    # Assemble Jacobian matrix
    fluxes = (dGjp12_dTj, dGjp12_dTjp1, dGjm12_dTj, dGjm12_dTjm1)
    signs = (-1, -1, +1, +1)
    offsets = (0, +1, 0, -1)
    for flux, sign, offset in zip(fluxes, signs, offsets):
        globflux = createGlobFlux(flux, len(Tnu), iStart, iEnd, offset)
        J += dia_matrix((sign * globflux, offset), shape=(len(Tnu), len(Tnu)))

    # Store first column in separate vector
    first_col[iStart:iEnd + 1] += -dGjp12_dsdot + dGjm12_dsdot

    return J, first_col


def addGridVector(fnu, Tnu, Tmap, rhonu, rhomap, layers, key):

    lay = layers[int(key[3:])]
    # The deeper layers are not moving, nor is the front layer if we don't have an ablative case
    # Thus we don't have any convection due to grid movement
    if int(key[3:]) != 0 and not lay.ablative:
        return fnu

    # Store some variables
    iStart = Tmap[key][0]
    iEnd = Tmap[key][-1]
    mat = lay.material
    gr = lay.grid
    sdot = Tnu[Tmap["sdot"]]
    Tj = Tnu[Tmap[key]]
    rhoj = rhonu[rhomap[key]]

    ### Grid movement ###
    # Flux at plus side
    Gjp12 = p12(rhoj * mat.e(Tj, lay.wv)) * gr.etajp12 * sdot
    # Flux at minus side
    Gjm12 = m12(rhoj * mat.e(Tj, lay.wv)) * gr.etajm12 * sdot
    Gjm12[0] = rhoj[0] * mat.e(Tj[0], lay.wv[0]) * sdot  # Account for boundaries

    # Assemble vector
    fnu[iStart:iEnd + 1] += -Gjp12 + Gjm12

    return fnu


def addPyroMatrix(J, Tnu, Tmap, mgas, layers, key, inputvars):

    lay = layers[int(key[3:])]
    # The deeper layers are not pyrolyzing, nor is the front layer if we don't have an ablative case
    # Thus we don't have any convection due to pyrolysis gas
    if int(key[3:]) != 0 and not lay.ablative:
        return J

    # Store some variables
    iStart = Tmap[key][0]
    iEnd = Tmap[key][-1]
    mat = lay.material
    #gr = lay.grid
    Tj = Tnu[Tmap[key]]

    # Calculate mass fluxes
    mgasm12 = mgas[::-1].cumsum()[::-1]
    mgasp12 = np.hstack((mgasm12[1:], 0))

    ### Pyrolysis gas convection ###
    # Flux at plus side
    dPjp12_dTj = 1/2 * mgasp12 * mat.gas.cp(Tj)
    dPjp12_dTjp1 = 1/2 * mgasp12 * p1(mat.gas.cp(Tj))
    # sensitivity to sdot is zero
    dPjp12_dTj[-1], dPjp12_dTjp1[-1] = (0, 0)  # impermeable back side

    # Flux at minus side
    dPjm12_dTj = 1/2 * mgasm12 * mat.gas.cp(Tj)
    dPjm12_dTjm1 = 1/2 * mgasm12 * m1(mat.gas.cp(Tj))
    if inputvars.BCfrontType == "heatflux":
        dPjm12_dTj[0], dPjm12_dTjm1[0] = (mgasm12[0] * mat.gas.cp(Tj[0]), 0)
    elif inputvars.BCfrontType == "aerodynamic":
        dPjm12_dTj[0], dPjm12_dTjm1[0] = (0, 0)  # Implemented in WallBlowMatrix
    else:
        raise ValueError("Unimplemented front BC %s" % inputvars.BCfrontType)

    # Assemble Jacobian matrix
    fluxes = (dPjp12_dTj, dPjp12_dTjp1, dPjm12_dTj, dPjm12_dTjm1)
    signs = (-1, -1, +1, +1)
    offsets = (0, +1, 0, -1)
    for flux, sign, offset in zip(fluxes, signs, offsets):
        globflux = createGlobFlux(flux, len(Tnu), iStart, iEnd, offset)
        J += dia_matrix((sign * globflux, offset), shape=(len(Tnu), len(Tnu)))

    return J


def addPyroVector(fnu, Tnu, Tmap, mgas, layers, key, inputvars):

    lay = layers[int(key[3:])]
    # The deeper layers are not pyrolyzing, nor is the front layer if we don't have an ablative case
    # Thus we don't have any convection due to pyrolysis gas
    if int(key[3:]) != 0 and not lay.ablative:
        return fnu

    # Store some variables
    iStart = Tmap[key][0]
    iEnd = Tmap[key][-1]
    mat = lay.material
    # gr = lay.grid
    Tj = Tnu[Tmap[key]]

    # Calculate mass fluxes
    mgasm12 = mgas[::-1].cumsum()[::-1]
    mgasp12 = np.hstack((mgasm12[1:], 0))

    ### Pyrolysis gas convection ###
    # Flux at plus side
    Pjp12 = mgasp12 * p12(mat.gas.h(Tj))
    Pjp12[-1] = 0  # impermeable back side

    # Flux at minus side
    Pjm12 = mgasm12 * m12(mat.gas.h(Tj))
    Pjm12[0] = mgasm12[0] * mat.gas.h(Tj[0])
    if inputvars.BCfrontType == "heatflux":
        Pjm12[0] = mgasm12[0] * mat.gas.h(Tj[0])
    elif inputvars.BCfrontType == "aerodynamic":
        Pjm12[0] = 0  # Implemented in WallBlowVector
    else:
        raise ValueError("Unimplemented front BC %s" % inputvars.BCfrontType)

    # Assemble Jacobian matrix
    fnu[iStart:iEnd + 1] += Pjp12 - Pjm12

    return fnu


def addWallBlowMatrix(J, first_col, Tnu, Tmap, rhonu, rhomap, mgas, layers, inputvars):

    # Store some variables
    iStart = Tmap["lay0"][0]
    lay = layers[0]
    mat = lay.material
    # gr = lay.grid
    sdot = Tnu[Tmap["sdot"]]
    Tw = Tnu[iStart]
    wvw = lay.wv[0]
    rhow = rhonu[rhomap["lay0"][0]]
    mg = np.sum(mgas)
    mc = rhow * sdot

    # Calculate corrected bg
    lam = 0.4 if inputvars.turbflow else 0.5
    phi = 2 * lam * (mg+mc) / inputvars.aerocoef
    blowcor = blowFromPhi(phi)
    bg = mg/(inputvars.aerocoef*blowcor)

    # Calculate sensitivity w.r.t. sdot
    dblowcor_dsdot = dblowdsdotFromPhi(phi, rhow, lam, inputvars.aerocoef)
    dhw_dsdot = mat.dhwdbg(bg, Tw) * mg/inputvars.aerocoef * dblowcor_dsdot / (-blowcor ** 2)
    dPw_dsdot = rhow * mat.hw(bg, Tw) + (mc + mg) * dhw_dsdot

    # Calculate sensitivity w.r.t. Tw
    dPw_dTw = (mc + mg) * mat.dhwdT(bg, Tw)

    # Assemble Jacobian matrix
    globflux = createGlobFlux(dPw_dTw, length=len(Tnu), iStart=iStart, iEnd=iStart, offset=0)
    J += dia_matrix((+1 * globflux, 0), shape=(len(Tnu), len(Tnu)))

    first_col[iStart] += dPw_dsdot

    return J, first_col


def addWallBlowVector(fnu, Tnu, Tmap, rhonu, rhomap, mgas, layers, inputvars):

    # Store some variables
    iStart = Tmap["lay0"][0]
    lay = layers[0]
    mat = lay.material
    # gr = lay.grid
    sdot = Tnu[Tmap["sdot"]]
    Tw = Tnu[iStart]
    # wvw = lay.wv[0]
    rhow = rhonu[rhomap["lay0"][0]]
    mg = np.sum(mgas)
    mc = rhow * sdot

    # Calculate corrected bg
    lam = 0.4 if inputvars.turbflow else 0.5
    phi = 2 * lam * (mg + mc) / inputvars.aerocoef
    blowcor = blowFromPhi(phi)
    bg = mg / (inputvars.aerocoef * blowcor)

    # Calculate wall blow flux
    Pw = (mc + mg) * mat.hw(bg, Tw)

    fnu[iStart] += Pw

    return fnu


def addHeatFluxVector(fnu, Tmap, inputvars, t):

    # Determine first cell
    iStart = Tmap["lay0"][0]
    fnu[iStart] += -inputvars.BCfrontValue(t)

    return fnu


def addAerodynamicMatrix(J, first_col, Tnu, Tmap, rhonu, rhomap, mgas, layers, inputvars):

    # Store some variables
    iStart = Tmap["lay0"][0]
    lay = layers[0]
    mat = lay.material
    # gr = lay.grid
    sdot = Tnu[Tmap["sdot"]]
    Tw = Tnu[iStart]
    # wvw = lay.wv[0]
    rhow = rhonu[rhomap["lay0"][0]]
    mg = np.sum(mgas)
    mc = rhow * sdot

    # Calculate corrected bg
    lam = 0.4 if inputvars.turbflow else 0.5
    phi = 2 * lam * (mg + mc) / inputvars.aerocoef
    blowcor = blowFromPhi(phi)
    bg = mg / (inputvars.aerocoef * blowcor)

    # Sensitivity to wall temperature
    dQw_dTw = inputvars.aerocoef * blowcor * (-mat.dhwdT(bg, Tw))

    # Sensitivity to recession rate
    dblowcor_dsdot = dblowdsdotFromPhi(phi, rhow, lam, inputvars.aerocoef)
    dhw_dsdot = mat.dhwdbg(bg, Tw) * mg / inputvars.aerocoef * dblowcor_dsdot / (-blowcor ** 2)
    dQw_dsdot = inputvars.aerocoef * (mat.hatmo(inputvars.BLEdgeT) - mat.hw(bg, Tw)) * dblowcor_dsdot + \
                inputvars.aerocoef * blowcor * (-dhw_dsdot)

    # Assemble Jacobian matrix
    globflux = createGlobFlux(dQw_dTw, length=len(Tnu), iStart=iStart, iEnd=iStart, offset=0)
    J += dia_matrix((-1 * globflux, 0), shape=(len(Tnu), len(Tnu)))

    first_col[iStart] += -dQw_dsdot

    return J, first_col


def addAerodynamicVector(fnu, Tnu, Tmap, rhonu, rhomap, mgas, layers, inputvars):

    # Store some variables
    iStart = Tmap["lay0"][0]
    lay = layers[0]
    mat = lay.material
    # gr = lay.grid
    sdot = Tnu[Tmap["sdot"]]
    Tw = Tnu[iStart]
    # wvw = lay.wv[0]
    rhow = rhonu[rhomap["lay0"][0]]
    mg = np.sum(mgas)
    mc = rhow * sdot

    # Calculate corrected bg
    lam = 0.4 if inputvars.turbflow else 0.5
    phi = 2 * lam * (mg + mc) / inputvars.aerocoef
    blowcor = blowFromPhi(phi)
    bg = mg / (inputvars.aerocoef * blowcor)

    # Calculate wall heat flux
    Qw = inputvars.aerocoef * blowcor * (mat.hatmo(inputvars.BLEdgeT) - mat.hw(bg, Tw))

    fnu[iStart] += -Qw

    return fnu


def addBcMatrix(J, first_col, Tnu, Tmap, rhonu, rhomap, mgas, layers, inputvars):

    # Store some variables
    isdot = Tmap["sdot"]
    iStart = Tmap["lay0"][0]
    lay = layers[0]
    mat = lay.material
    # gr = lay.grid
    sdot = Tnu[Tmap["sdot"]]
    Tw = Tnu[iStart]
    # wvw = lay.wv[0]
    rhow = rhonu[rhomap["lay0"][0]]
    mg = np.sum(mgas)
    mc = rhow * sdot

    # Calculate corrected bg
    lam = 0.4 if inputvars.turbflow else 0.5
    phi = 2 * lam * (mg + mc) / inputvars.aerocoef
    blowcor = blowFromPhi(phi)
    bg = mg / (inputvars.aerocoef * blowcor)

    # Sensitivity to recession rate
    dblowcor_dsdot = dblowdsdotFromPhi(phi, rhow, lam, inputvars.aerocoef)
    dbg_dsdot = mg / inputvars.aerocoef * dblowcor_dsdot / (-blowcor ** 2)
    dR0_dsdot = rhow / (inputvars.aerocoef * blowcor) + \
                rhow * sdot / (inputvars.aerocoef * (-blowcor**2)) * dblowcor_dsdot - \
                mat.dbcdbg(bg, Tw) * dbg_dsdot

    # Sensitivity to temperature
    dR0_dTw = -mat.dbcdT(bg, Tw)

    # Assemble Jacobian matrix
    globflux = np.zeros(len(Tnu))
    globflux[iStart] = dR0_dTw
    J += dia_matrix((+1 * globflux, +1), shape=(len(Tnu), len(Tnu)))

    first_col[isdot] += dR0_dsdot

    return J, first_col


def addBcVector(fnu, Tnu, Tmap, rhonu, rhomap, mgas, layers, inputvars):

    # Store some variables
    isdot = Tmap["sdot"]
    iStart = Tmap["lay0"][0]
    lay = layers[0]
    mat = lay.material
    # gr = lay.grid
    sdot = Tnu[Tmap["sdot"]]
    Tw = Tnu[iStart]
    # wvw = lay.wv[0]
    rhow = rhonu[rhomap["lay0"][0]]
    mg = np.sum(mgas)
    mc = rhow * sdot

    # Calculate corrected bg
    lam = 0.4 if inputvars.turbflow else 0.5
    phi = 2 * lam * (mg + mc) / inputvars.aerocoef
    blowcor = blowFromPhi(phi)
    bg = mg / (inputvars.aerocoef * blowcor)

    # Calculate aerodynamic heating flux
    R0 = rhow * sdot / (inputvars.aerocoef * blowcor) - mat.bc(bg, Tw)

    fnu[isdot] += R0

    return fnu


def blowFromPhi(phi):
    if phi < 1.0e-7:
        return 1 - phi/2 + (phi**2)/12
    elif phi < 20:
        return phi/(np.e**phi-1)
    elif phi < 100:
        return phi * np.e**(-phi)
    else:
        return 1.0e-8


def dblowdsdotFromPhi(phi, rhow, lam, aerocoef):
    dphi_dsdot = 2 * lam * rhow / aerocoef

    if phi < 1.0e-7:
        return 0.5 * (phi/3 - 1) * dphi_dsdot
    elif phi < 20:
        return (((np.e**phi) * (1-phi) -1)/((np.e**phi)-1)**2) * dphi_dsdot
    elif phi < 100:
        return (np.e**(-phi)) * (1-phi) * dphi_dsdot
    else:
        return 0


def updateRho(lay, rhoimu, rhoin, Tnu, Tmap, tDelta):

    key = "lay0"
    Tj = Tnu[Tmap[key]]
    mat = lay.material
    gr = lay.grid

    #Tj = np.linspace(100, 900, len(Tj))  # TODO: remove later, only for debugging purposes

    iteration = 0

    while True:

        iteration += 1

        deltaRhoimu = ((rhoimu - rhoin - tDelta * drhodt(mat, rhoimu, Tj)) /
                       (tDelta * ddrhodt_drho(mat, rhoimu, Tj) - np.ones(rhoimu.shape)))
        rhoimu += deltaRhoimu
        if np.linalg.norm(deltaRhoimu/rhoimu) < 1.0e-8:
            print("Rho determination completed after %i iterations." % iteration)
            break

    frac = np.repeat(mat.data.frac.reshape(1, -1), repeats=len(Tj), axis=0)
    rhonu = np.sum(frac*rhoimu, axis=1)

    rhv, rhc = (mat.data.rhov0, mat.data.rhoc0)
    lay.wv = rhv/(rhv-rhc) * (1-rhc/rhonu)

    mgas = -np.sum(frac*drhodt(mat, rhoimu, Tj), axis=1) * (gr.zjp12-gr.zjm12)

    return rhonu, rhoimu, mgas


def drhodt(mat, rhoi, Tj):
    rhoc = np.repeat(mat.data.charRhoFrac0.reshape(1, -1), repeats=len(Tj), axis=0)
    rhov = np.repeat(mat.data.virginRhoFrac0.reshape(1, -1), repeats=len(Tj), axis=0)
    kr = np.repeat(mat.data.kr.reshape(1, -1), repeats=len(Tj), axis=0)
    nr = np.repeat(mat.data.nr.reshape(1, -1), repeats=len(Tj), axis=0)
    Ei = np.repeat(mat.data.Ei.reshape(1, -1), repeats=len(Tj), axis=0)
    T = np.repeat(Tj.reshape(-1,1), repeats=len(mat.data.charRhoFrac0), axis=1)
    decomp = rhoc != rhov
    val = np.zeros(rhoi.shape)
    val[decomp] = -kr[decomp] * ((rhoi[decomp] - rhoc[decomp]) / (rhov[decomp] - rhoc[decomp])) ** nr[decomp] \
                  * np.exp(-Ei[decomp] / T[decomp])
    return val


def ddrhodt_drho(mat, rhoi, Tj):
    rhoc = np.repeat(mat.data.charRhoFrac0.reshape(1, -1), repeats=len(Tj), axis=0)
    rhov = np.repeat(mat.data.virginRhoFrac0.reshape(1, -1), repeats=len(Tj), axis=0)
    kr = np.repeat(mat.data.kr.reshape(1, -1), repeats=len(Tj), axis=0)
    nr = np.repeat(mat.data.nr.reshape(1, -1), repeats=len(Tj), axis=0)
    Ei = np.repeat(mat.data.Ei.reshape(1, -1), repeats=len(Tj), axis=0)
    T = np.repeat(Tj.reshape(-1, 1), repeats=len(mat.data.charRhoFrac0), axis=1)
    decomp = rhoc != rhov
    val = np.zeros(rhoi.shape)
    val[decomp] = -kr[decomp] * nr[decomp] / (rhov[decomp] - rhoc[decomp]) *\
                  ((rhoi[decomp] - rhoc[decomp]) / (rhov[decomp] - rhoc[decomp])) ** (nr[decomp] - 1) \
                  * np.exp(-Ei[decomp] / T[decomp])
    return val