from distutils.command.bdist import bdist

import calfem.geometry as cfg
import calfem.mesh as cfm

import calfem.core as cfc
import calfem.utils as cfu

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use('TkAgg')

import calfem.vis_mpl as cfv

import numpy as np

# Mesh data
el_sizef, el_type= 0.03, 2 # el_size = 0.08 ser nice ut, el_type vet ej vad den gör
thickness = 1 # meter
mesh_dir = "./"

MARKER_CuCr = 0
ECu = 139       # Young's modulus (GPa)
VCu = 0.18      # Poisson's ratio
AlphaCu = 17e-6 # thermal expansion coefficient (1/K)
RhoCu = 8890    # Density (kg/m^3)
CCu = 377       # Specific heat (J/KgK)
KCu = 323       # Thermal conductivity (W/mK)

MARKER_TiAlloy = 1
ETi = 108       # Young's modulus (GPa)
VTi = 0.22      # Poisson's ratio
AlphaTi = 8.8e-6# thermal expansion coefficient (1/K)
RhoTi = 4540    # Density (kg/m^3)
CTi = 590       # Specific heat (J/KgK)
KTi = 12        # Thermal conductivity (W/mK)

MARKER_Inside = 2
ChamberHeating = 5000 # W/m^2
InsidePressure = 1e6 # Pa

MARKER_BellOutside = 3
AlphaConvection = 50 # W/m^2K
Tinfty = 293 # K
T0 = Tinfty

MARKER_ChamberOutside = 4
ChamberCooling = 1000 # W/m^2

MARKER_QN_0 = 5
MARKER_Material_Transition = 6 # vet inte om något särskillt behöver göras, men verkar inte så



def nodesToEdges ( nodes : dict , enod : np . array ) -> dict :
    """ Returns a list of edges given nodes
    Args:
    nodes: nodes on boundary, bdofs
    enod: element connectivity matrix, edofs

    returns a list of edges on the boundary
    """
    edges = {}
    for key in nodes.keys() :
        edges[key] = []

    for con in zip(enod) :
        for key in nodes.keys():
            I = np.intersect1d(con, nodes[key])
            if len(I) == 2 :
                edges[key].append(I)
    return edges

def plantml(ex: np.array, ey: np.array, s: float):
    """
    Computes the integral of the form-functions over a 3-node triangle element
        Me = int(s*N^T*N)dA

    Inputs:
        ex: element x-coordinates
        ey: element y-coordinates
        s: constant scalar, e.g. density*thickness

    Outputs:
        Me: integrated element matrix
    """
    if not ex.shape == (3,) or not ey.shape == (3,):
        raise Exception("Incorrect shape of ex or ey: {0}, {1} but should be(3,)".format(ex.shape, ey.shape))
    # Compute element area
    Cmat = np.vstack((np.ones((3, )), ex, ey))
    A = np.linalg.det(Cmat)/2
    # Set up quadrature
    g1 = [0.5, 0.0, 0.5]
    g2 = [0.5, 0.5, 0.0]
    g3 = [0.0, 0.5, 0.5]
    w = (1/3)
    # Perform numerical integration
    Me = np.zeros((3, 3))
    for i in range(0, 3):
        Me += w*np.array([
            [g1[i]**2, g1[i]*g2[i], g1[i]*g3[i]],
            [g2[i]*g1[i], g2[i]**2, g2[i]*g3[i]],
            [g3[i]*g1[i], g3[i]*g2[i], g3[i]**2]])
    Me *= A*s
    return Me

def NozzleGeom() :
    g = cfg.geometry()
    g.point([0, 0], 0)
    g.point([0, 0.5], 1)
    g.point([1.4, 0.5], 12)  # material changes here
    g.point([3.5 - 1.7 * np.cos(np.arcsin(0.5 / 1.1)), 0.5], 2)
    g.point([3.5, 1.1], 3)  # top of ellipse
    g.point([3.5, 0.9], 4)  # inside of top of ellipse
    g.point([3.5 - 1.5 * np.cos(np.arcsin(0.2 / 0.9)), 0.2], 5)
    g.point([3.5, 0], 6)  # center of ellipse
    g.point([1.4 + np.sqrt(0.3 * 0.3 - 0.2 * 0.2), 0.2], 7)
    g.point([1.4, 0.3], 8)
    g.point([0.7, 0.3], 9)
    g.point([0.7, 0], 10)
    g.point([1.4, 0], 11)

    # 1.7 cos(v) x + 1.1 sin(v) y, v:y=0.2 => sin(v) = 0.2/1.1 => v = arcsin(0.2/1.1) => x = 1.7 * cos(arcsin(0.2/1.1))

    g.spline([0, 1], 0, marker=MARKER_ChamberOutside)
    g.spline([1, 12], 1, marker=MARKER_ChamberOutside)
    g.spline([12, 2], 10, marker=MARKER_BellOutside)
    g.ellipse([2, 6, 11, 3], 2, marker=MARKER_BellOutside)  # [startpoint, centerpoint, mAxisPoint, endpoint]
    g.spline([3, 4], 3, marker=MARKER_BellOutside)
    g.ellipse([5, 6, 11, 4], 4, marker=MARKER_Inside)
    g.spline([5, 7], 5, marker=MARKER_Inside)
    g.circle([7, 11, 8], 6, marker=MARKER_Inside)
    g.spline([8, 9], 7, marker=MARKER_Inside)
    g.spline([9, 10], 8, marker=MARKER_Inside)
    g.spline([10, 0], 9, marker=MARKER_QN_0)
    g.spline([12, 8], 11, marker=MARKER_Material_Transition)

    g.surface([0, 1, 11, 7, 8, 9], ID=0, marker=MARKER_TiAlloy)
    g.surface([10, 2, 3, 4, 5, 6, 11], ID=1, marker=MARKER_CuCr)
    return g

def MakeThermMesh(geom) :
    mesh = cfm.GmshMeshGenerator(geom, mesh_dir=mesh_dir)

    dofs_pn = 1 # temp
    mesh.el_size_factor = el_sizef
    mesh.el_type = el_type
    mesh.dofs_per_node = dofs_pn

    coord, edof, dofs, bdofs, element_markers = mesh.create()

    return (coord, edof, dofs, bdofs, element_markers)

def AssembleThermStiffness(coord, edof, dofs, bdofs, element_markers) :
    epCu = np.array([RhoCu])
    epTi = np.array([RhoTi])
    n_dofs = np.size(dofs)
    ex, ey = cfc.coordxtr(edof, coord, dofs)
    DCu = np.array([[KCu, 0], [0, KCu]])  # Thermal transfer constitutive matrix
    DTi = np.array([[KTi, 0], [0, KTi]])  # Thermal transfer constitutive matrix
    K = np.zeros([n_dofs, n_dofs])

    for i in range(len(edof)):
        if element_markers[i] == 1:
            Ke = cfc.flw2te(ex[i], ey[i], epTi, DTi)
        else:
            Ke = cfc.flw2te(ex[i], ey[i], epCu, DCu)
        K = cfc.assem(edof[i], K, Ke)

    return K

def MakeThermBC(F, bdofs) :
    edges = nodesToEdges(bdofs, edof)

    bc, bc_value = np.array([], 'i'), np.array([], 'f')
    #bc, bc_value = cfu.applybc(bdofs, bc, bc_value, MARKER_QN_0, 0, 0)
    #bc, bc_value = cfu.applybc(bdofs, bc, bc_value, MARKER_BellOutside, 0, 0)
    #bc, bc_value = cfu.applybc(bdofs, bc, bc_value, MARKER_Inside, 1000, 0)
    #bc, bc_value = cfu.applybc(bdofs, bc, bc_value, MARKER_ChamberOutside, 0, 0)

    # Add heating BC to inside of rocket
    for e in edges[MARKER_Inside] :
        #print(e)
        x1, y1 = coord[e[0]-1]
        #print("x1, y1: ", x1, y1)
        x2, y2 = coord[e[1]-1]
        #print("x2, y2: ", x2, y2)
        dx = x1-x2
        dy = y1-y2
        l = np.sqrt(dx*dx + dy*dy)
        #print("l: ", l)
        F[e[0]-1] += l*thickness*ChamberHeating/2
        F[e[1]-1] += l*thickness*ChamberHeating/2

    # Add cooling BC to outside of nozzle
    for e in edges[MARKER_ChamberOutside] :
        x1, y1 = coord[e[0]-1]
        x2, y2 = coord[e[1]-1]
        dx = x1-x2
        dy = y1-y2
        l = np.sqrt(dx*dx + dy*dy)
        F[e[0]-1] -= l*thickness*ChamberCooling/2
        F[e[1]-1] -= l*thickness*ChamberCooling/2


    # Add convecton BC
    KModifier = np.zeros([len(F), len(F)])
    for e in edges[MARKER_BellOutside] :
        x1, y1 = coord[e[0]-1]
        x2, y2 = coord[e[1]-1]
        dx = x1-x2
        dy = y1-y2
        l = np.sqrt(dx*dx + dy*dy)
        F[e[0]-1] += l*thickness*AlphaConvection*Tinfty/2
        F[e[1]-1] += l*thickness*AlphaConvection*Tinfty/2
        KModifier[e[0]-1][e[0]-1] += l * thickness * AlphaConvection /2
        KModifier[e[1]-1][e[1]-1] += l * thickness * AlphaConvection /2

    #print(F)

    return F, bc, bc_value, KModifier

def plotTherm(a) :
    UNIT = 1 / 2.54
    wcm, hcm = 35, 10
    fig, ax = plt.subplots(figsize=(wcm * UNIT, hcm * UNIT))

    v = np.asarray(a)
    x, y = coord.T
    edof_tri = cfv.topo_to_tri(edof)
    tri = plt.tripcolor(x, y, edof_tri - 1, v.ravel(), shading="gouraud")

    fig.colorbar(tri, ax=ax, label='Temperature [K]')
    plt.title("Temperature distribution at equilibrium")

    plt.show()


def MakeMechMesh(geom) :
    mesh = cfm.GmshMeshGenerator(geom, mesh_dir=mesh_dir)

    dofs_pn = 2 # skutning x, skjutning y

    mesh.el_size_factor = el_sizef
    mesh.el_type = el_type
    mesh.dofs_per_node = dofs_pn

    coord, edof, dofs, bdofs, element_markers = mesh.create()

    """# Plotta den generade meshen
    fig, ax = plt.subplots()
    cfv.draw_geometry(
        geom,
        label_curves=True,
        title="Geometry: Computer Lab Exercise 2"
    )
    plt.show()
    cfv.draw_mesh(coords=coord, edof=edof, dofs_per_node=mesh.dofs_per_node, el_type=mesh.el_type, filled=True)
    plt.show()"""

    return (coord, edof, dofs, bdofs, element_markers)

def AssembleMechStiffness(coord, edof, dofs, bdofs, element_markers) :
    # Assemble plane strain matrix
    ptype = 1
    ep = np.array([ptype, 1])
    n_dofs = np.size(dofs)
    ex, ey = cfc.coordxtr(edof, coord, dofs)
    DCu = cfc.hooke(ptype, ECu, VCu)
    DTi = cfc.hooke(ptype, ETi, VTi)
    K = np.zeros([n_dofs, n_dofs])

    for i in range(len(edof)):
        if element_markers[i] == 1:
            Ke = cfc.plante(ex[i], ey[i], ep, DTi)
        else:
            Ke = cfc.plante(ex[i], ey[i], ep, DCu)
        K = cfc.assem(edof[i], K, Ke)

    return K

def MakeMechBC(F, bdofs) :
    edges = nodesToEdges(bdofs, edof)

    bc, bc_value = np.array([], 'i'), np.array([], 'f')
    bc, bc_value = cfu.applybc(bdofs, bc, bc_value, MARKER_ChamberOutside, 0, 0)



if __name__=="__main__":

    # Solve stationary thermal problem
    coord, edof, dofs, bdofs, element_markers = MakeThermMesh(NozzleGeom())

    K = AssembleThermStiffness(coord, edof, dofs, bdofs, element_markers)

    F = np.zeros([np.size(dofs), 1])

    F, bc, bc_value, KModifier = MakeThermBC(F, bdofs)

    K = K+KModifier # Update K with changes from convection BC

    # Solve thermal problem
    a, r = cfc.solveq(K, F, bc, bc_value)

    if True :   # Swich to turn of plotting for thermal problem
        fig, ax = plt.subplots()
        cfv.draw_geometry(
            NozzleGeom(),
            label_curves=True,
            title="Geometry: Computer Lab Exercise 2"
        )
        plt.show()
        cfv.draw_mesh(coords=coord, edof=edof, dofs_per_node=1, el_type=2, filled=True)
        plt.show()

        plotTherm(a)


    # Solve stationary mechanical problem       Nevermind, don't really care about that right now
    coord, edof, dofs, bdofs, element_markers = MakeMechMesh(NozzleGeom())

    K = AssembleMechStiffness(coord, edof, dofs, bdofs, element_markers)

    F = np.zeros([np.size(dofs), 1])

    cfu.applyforce(bdofs, F, MARKER_Inside, 1000, 0)

    #F, bc, bc_value = MakeMechBC(F, bdofs)





    # Eventuellt behöver även bc för tid = 0 sättas


    """SteadyStateSim()
    SteadyStatePlot()
    TransientSim()      # cfc.step1() kan vara användbart
    TransientPlot()
    VonMisesSim()       # se https://calfem-for-python.readthedocs.io/en/latest/examples/exm10.html
    VonMisesPlot()      # cfc.plante() kan vara användbart, se AssembleMechStiffness()"""