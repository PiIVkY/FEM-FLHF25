import calfem.core as cfc
import numpy as np


def MaltSix(nodes_in_el, eltopo, temps, T_0, E, v, Alpha, ep, elx, ely, f_mechanical) :
    el_D_matrix = D_matrix(E, v)
    el_nu = v
    el_alpha = Alpha
    el_E = E
    K_mechanical = np.zeros((6,6))

    # Average out the nodal temperatures to get a mean temperature for the element,
    # since the integral is over the entire element
    meanT = 1 / 3 * (temps[nodes_in_el[0] - 1] + temps[nodes_in_el[1] - 1]
                     + temps[nodes_in_el[2] - 1])[0]
    # Calculates the temperature difference from the
    deltaT = meanT - T_0
    print("dT: ", deltaT)

    # Element contribution due to thermal expansion
    epsilon_deltaT = el_alpha * deltaT * np.array([[1], [1], [0]])
    es = el_D_matrix @ epsilon_deltaT  # @ Ser läskig ut men är bara matrismultiplikation

    print("es: ", es.T)

    fe = cfc.plantf(elx, ely, ep, es.T)

    print("fe: ", fe)

    # el_D_matrix = cfc.hooke(2, el_E, el_nu)
    Ke = cfc.plante(elx, ely, ep, el_D_matrix)
    cfc.assem(eltopo, K_mechanical, Ke, f_mechanical, fe)
    print("f: ", f_mechanical)

def D_matrix(E, v):
    D_mat = E / ((1 + v) * (1 - 2 * v)) * np.array([[1 - v, v, 0],
                                                    [v, 1 - v, 0],
                                                    [0, 0, 1 / 2 * (1 - 2 * v)]])
    return D_mat



def JohAnt(nodes_in_el, edof, temps, Tinfty, ETi, VTi, AlphaTi, ep, ex, ey, F) :
    es = np.array([[0, 0, 0]])

    # Average temp of element
    dof = edof[0]
    temp = (temps[int(dof[1] / 2 - 1)] + temps[int(dof[3] / 2 - 1)] + temps[int(dof[5] / 2 - 1)]) / 3
    temp = temp[0] - Tinfty
    print("dT: ", temp)

    preSigTi = np.array([[1, 1, 0]]) * AlphaTi * ETi / (1 - 2 * VTi)
    #preSigCu = np.array([[1, 1, 0]]) * AlphaCu * ECu / (1 - 2 * VCu)

    #if element_markers[i] == MARKER_TiAlloy:
    es[0] = preSigTi * temp


    es =

    print("es: ", es)

    #else:
    #    es[0] = preSigCu * temp

    Fe = cfc.plantf(ex, ey, ep, es)
    print("fe: ", Fe)

    F = customFAssm(edof, F, Fe)
    print("f: ", F)

def customFAssm(edof, f, fe) :
    for row in edof:
        idx = row - 1
        f[np.ix_(idx)] = f[np.ix_(idx)] + fe
    return f

def customHooke(E, v) -> np.matrix:
    D = E * np.matrix(
        [[1-v, v, 0],
         [v, 1-v, 0],
         [0, 0, (1 - 2 * v) / 2]]
    ) / ((1 + v) * (1 - 2 * v))
    return D

if __name__=="__main__":
    edof = np.array([[1, 2, 3, 4, 5, 6]])
    nodes_in_el = [1, 2, 3]
    temps = np.transpose([[301, 302, 303]])
    Tinfty = 300
    ETi = 100e9
    VTi = 0.2
    AlphaTi = 1e-5
    ep = [2, 1]
    ex = [0, 1, 0]
    ey = [0, 0, 1]
    F = np.zeros((6,1))

    print("Malte och Sixtens kod: ")
    MaltSix(nodes_in_el, edof, temps, Tinfty, ETi, VTi, AlphaTi, ep, ex, ey, F)
    print("")
    print("Johan och Antons kod: ")
    JohAnt(nodes_in_el, edof, temps, Tinfty, ETi, VTi, AlphaTi, ep, ex, ey, F)