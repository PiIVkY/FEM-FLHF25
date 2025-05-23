import calfem.geometry as cfg
import calfem.mesh as cfm

import calfem.core as cfc
import calfem.utils as cfu

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation

mpl.use('TkAgg')

import calfem.vis_mpl as cfv

import numpy as np


# Mesh data
el_sizef, el_type= 0.08, 2 # el_size = 0.08 ser nice ut, el_type vet ej vad den gör

mesh_dir = "./"

MARKER_CuCr = 0

MARKER_TiAlloy = 1
ETi = 108e9     # Young's modulus (Pa)
VTi = 0.22      # Poisson's ratio
AlphaTi = 8.8e-6# thermal expansion coefficient (1/K)
RhoTi = 4540    # Density (kg/m^3)
CTi = 590       # Specific heat (J/KgK)
KTi = 12        # Thermal conductivity (W/mK)

MARKER_Inside = 2
ChamberHeating = 5000 # W/m^2

MARKER_BellOutside = 3

MARKER_ChamberOutside = 4
ChamberCooling = 1000 # W/m^2

MARKER_QN_0 = 5
MARKER_Material_Transition = 6 # vet inte om något särskillt behöver göras, men verkar inte så
MARKER_TCONST = 7
TCONST = 100 # K



def test(plot) :
    some_constants = {
        "AlphaConvection" : 1000,
        "thickness" : 0.010,  # m
        "KCu" : 10,
        "CCu": 100,  # Specific heat (J/KgK), Not in the test case but it feels good to include a test for the dynamic thermal problem
        "RhoCu" : 1,
        "ECu" : 200e9,
        "VCu" : 0.3,
        "Tinfty" : 20,  # C
        "InsidePressure" : -100e6,  # Pa
    }

    coord, edof, dofs, bdofs, element_markers = MakeThermTestMesh()
    K = AssembleThermStiffness(coord, edof, dofs, bdofs, element_markers, some_constants)

    F = np.zeros([np.size(dofs), 1])

    F, bc, bc_value, KModifier = MakeThermBC(F, bdofs, edof, coord, some_constants)
    K = K + KModifier  # Update K with changes from convection BC

    # Solve stationary thermal problem
    a, r = cfc.solveq(K, F, bc, bc_value)

    # Plot solution to steady state problem
    if plot:  # Swich to turn of plotting for thermal problem
        cfv.draw_mesh(coords=coord, edof=edof, dofs_per_node=1, el_type=2, filled=True)
        plt.show()

        print("Temperatures:")
        print(a)
        plotTherm(a, coord, edof)

    # Solve dynamic thermal problem
    #C = MakeCapacityMatrix(dofs, element_markers)
    C = MakeCapacityMatrix2(dofs, edof, element_markers, some_constants)

    a0 = np.ones((len(dofs), 1)) * some_constants["Tinfty"]

    dt, tottime, alpha = 10, 3600, 1

    smoothness = 0.5
    times = [100*i*smoothness for i in range(int(tottime/(100*smoothness)))]

    dynbc = np.concatenate((np.transpose([bc]), np.transpose([bc_value])), 1)

    modhist, dofhist = cfc.step1(K, C, F, a0, dynbc, [dt, tottime, alpha], times, dofs=np.array([]))

    # Plot solution to dynamic thermal problem
    if plot:
        UNIT = 1 / 2.54
        wcm, hcm = 35, 10
        fig, (ax, cbax) = plt.subplots( 1, 2, width_ratios=[10, 1], figsize=(wcm * UNIT, hcm * UNIT))

        x, y = coord.T
        fmt = '%1.2f'
        v = np.asarray(modhist["a"].transpose()[0])
        edof_tri = cfv.topo_to_tri(edof)
        im = ax.tripcolor(x, y, edof_tri - 1, v.ravel(), shading="gouraud")
        fig.colorbar(im, ax=ax, cax = cbax, label='Temperature [K]', format=fmt)
        #im.set_clim(Tinfty, 420)
        i0 = 0
        tx = ax.text(3, 0.1, str(i0))

        def animate(i):
            v = np.asarray(modhist["a"].transpose()[i])
            #vmax = np.max(v)
            #vmin = np.min(v)
            im = ax.tripcolor(x, y, edof_tri - 1, v.ravel(), shading="gouraud")
            fig.colorbar(im, ax=ax, cax = cbax, label='Temperature [K]', format=fmt)
            #im.set_clim(vmin, 420)
            tx.set_text("frame: " + str(i))

        plt.title("Temperature distribution at equilibrium")

        ani = animation.FuncAnimation(fig=fig, func=animate, frames = len(times), interval=(smoothness*200))
        plt.show()



    coord, edof, dofs, bdofs, element_markers = MakeMechTestMesh()

    K, DCu, DTi, ex, ey, ep = AssembleMechStiffness(coord, edof, dofs, bdofs, element_markers, some_constants)

    F = np.zeros([np.size(dofs), 1])

    F, bc, bc_value = MakeMechBC(F, coord, dofs, bdofs, edof, some_constants)

    F = F*-1

    a, r = cfc.solveq(K, F, bc, bc_value)

    # Calculate von mises stress
    ed = cfc.extract_eldisp(edof, a)  # element displacement

    von_mises = []

    for i in range(edof.shape[0]):
        if element_markers[i] == MARKER_CuCr:
            es, et = cfc.plants(ex[i, :], ey[i, :], ep, DCu, ed[i, :])
        else:
            es, et = cfc.plants(ex[i, :], ey[i, :], ep, DTi, ed[i, :])
        von_mises.append(np.sqrt(pow(es[0, 0], 2) - es[0, 0] * es[0, 1] + pow(es[0, 1], 2) + 3 * pow(es[0, 2], 2)))

    if plot :
        print("Displacements")
        print(a)
        cfv.figure(fig_size=(10, 5))
        cfv.draw_element_values(
            von_mises,
            coord,
            edof,
            2,
            el_type,
            a,
            draw_elements=False,
            draw_undisplaced_mesh=True,
            title="Effective stress and displacement",
            magnfac=100.0,
        )
        cfv.show_and_wait()

def statTherm(plot) :
    some_constants = {
        "AlphaConvection" : 50,  # W/m^2K
        "thickness" : 1,  # meter
        "ECu" : 139e9,  # Young's modulus (Pa)
        "VCu" : 0.18,  # Poisson's ratio
        "AlphaCu" : 17e-6,  # thermal expansion coefficient (1/K)
        "RhoCu" : 8890,  # Density (kg/m^3)
        "CCu" : 377,  # Specific heat (J/KgK)
        "KCu" : 323,  # Thermal conductivity (W/mK)
        "Tinfty": 293,  # K
        "InsidePressure": 1e6,  # Pa
    }


    coord, edof, dofs, bdofs, element_markers = MakeThermMesh(NozzleGeom())

    K = AssembleThermStiffness(coord, edof, dofs, bdofs, element_markers, some_constants)

    F = np.zeros([np.size(dofs), 1])

    F, bc, bc_value, KModifier = MakeThermBC(F, bdofs, edof, coord, some_constants)
    K = K + KModifier  # Update K with changes from convection BC

    # Solve stationary thermal problem
    a, r = cfc.solveq(K, F, bc, bc_value)

    # Plot solution to steady state problem
    if plot:  # Swich to turn of plotting for thermal problem
        cfv.draw_geometry(
            NozzleGeom(),
            label_curves=True,
            title="Geometry: Computer Lab Exercise 2"
        )
        plt.show()
        cfv.draw_mesh(coords=coord, edof=edof, dofs_per_node=1, el_type=2, filled=True)
        plt.show()
        plotTherm(a, coord, edof)

    return a

def dynTherm(plot) :
    some_constants = {
        "AlphaConvection": 50,  # W/m^2K
        "thickness": 1,  # meter
        "ECu": 139e9,  # Young's modulus (Pa)
        "VCu": 0.18,  # Poisson's ratio
        "AlphaCu": 17e-6,  # thermal expansion coefficient (1/K)
        "RhoCu": 8890,  # Density (kg/m^3)
        "CCu": 377,  # Specific heat (J/KgK)
        "KCu": 323,  # Thermal conductivity (W/mK)
        "Tinfty": 293,  # K
        "InsidePressure": 1e6,  # Pa
    }

    coord, edof, dofs, bdofs, element_markers = MakeThermMesh(NozzleGeom())

    K = AssembleThermStiffness(coord, edof, dofs, bdofs, element_markers, some_constants)

    F = np.zeros([np.size(dofs), 1])

    F, bc, bc_value, KModifier = MakeThermBC(F, bdofs, edof, coord, some_constants)
    K = K + KModifier  # Update K with changes from convection BC

    C = MakeCapacityMatrix(dofs, element_markers, some_constants)
    #C = MakeCapacityMatrix2(dofs, element_markers)

    a0 = np.ones((len(dofs), 1)) * some_constants["Tinfty"]

    dt, tottime, alpha = 10, 3600, 1

    smoothness = 0.5
    times = [100 * i * smoothness for i in range(int(tottime / (100 * smoothness)))]

    modhist, dofhist = cfc.step1(K, C, F, a0, bc, [dt, tottime, alpha], times, dofs=np.array([]))

    if plot:
        UNIT = 1 / 2.54
        wcm, hcm = 35, 10
        fig, (ax, cbax) = plt.subplots( 1, 2, width_ratios=[10, 1], figsize=(wcm * UNIT, hcm * UNIT))

        x, y = coord.T
        fmt = '%1.2f'
        v = np.asarray(modhist["a"].transpose()[0])
        edof_tri = cfv.topo_to_tri(edof)
        im = ax.tripcolor(x, y, edof_tri - 1, v.ravel(), shading="gouraud")
        fig.colorbar(im, ax=ax, cax = cbax, label='Temperature [K]', format=fmt)
        #im.set_clim(Tinfty, 420)
        i0 = 0
        tx = ax.text(3, 0.1, str(i0))

        def animate(i) :
            v = np.asarray(modhist["a"].transpose()[i])
            #vmax = np.max(v)
            #vmin = np.min(v)
            im = ax.tripcolor(x, y, edof_tri - 1, v.ravel(), shading="gouraud")
            fig.colorbar(im, ax=ax, cax = cbax, label='Temperature [K]', format=fmt)
            #im.set_clim(vmin, 420)
            tx.set_text("frame: " + str(i))


        plt.title("Temperature distribution at equilibrium")


        ani = animation.FuncAnimation(fig=fig, func=animate, frames = len(times), interval=(smoothness*200))
        plt.show()

def Mech(plot, temps) :
    some_constants = {
        "AlphaConvection": 50,  # W/m^2K
        "thickness": 1,  # meter
        "ECu": 139e9,  # Young's modulus (Pa)
        "VCu": 0.18,  # Poisson's ratio
        "AlphaCu": 17e-6,  # thermal expansion coefficient (1/K)
        "RhoCu": 8890,  # Density (kg/m^3)
        "CCu": 377,  # Specific heat (J/KgK)
        "KCu": 323,  # Thermal conductivity (W/mK)
        "InsidePressure": 1e6,  # Pa
    }

    # Solve stationary mechanical problem
    coord, edof, dofs, bdofs, element_markers = MakeMechMesh(NozzleGeom())

    K, DCu, DTi, ex, ey, ep = AssembleMechStiffness(coord, edof, dofs, bdofs, element_markers, some_constants)

    F = np.zeros([np.size(dofs), 1])

    F, bc, bc_value = MakeMechBC(F, coord, dofs, bdofs, edof, some_constants)

    a, r = cfc.solveq(K, F, bc, bc_value)

    # Calculate von mises stress
    ed = cfc.extract_eldisp(edof, a)  # element displacement

    von_mises = []

    for i in range(edof.shape[0]):
        if element_markers[i] == MARKER_CuCr:
            es, et = cfc.plants(ex[i, :], ey[i, :], ep, DCu, ed[i, :])
        else:
            es, et = cfc.plants(ex[i, :], ey[i, :], ep, DTi, ed[i, :])
        von_mises.append(np.sqrt(pow(es[0, 0], 2) - es[0, 0] * es[0, 1] + pow(es[0, 1], 2) + 3 * pow(es[0, 2], 2)))

    x = 0
    for i in von_mises:
        if (i > x):
            x = i
    print("biggest von_mises:", x / 1e6)

    cfv.figure(fig_size=(10, 5))
    cfv.draw_element_values(
        von_mises,
        coord,
        edof,
        2,
        el_type,
        a,
        draw_elements=False,
        draw_undisplaced_mesh=True,
        title="Effective stress and displacement",
        magnfac=10.0,
    )
    cfv.show_and_wait()




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

def MakeThermTestMesh() :
    coord = np.zeros((9, 2))
    coord[0] = [0, 0]           # Bottom left
    coord[1] = [0.005, 0]       # Bottom middle
    coord[2] = [0.01, 0]        # Bottom right
    coord[3] = [0, 0.005]       # Middle left
    coord[4] = [0.005, 0.005]   # Center
    coord[5] = [0.01, 0.005]    # Middle right
    coord[6] = [0, 0.01]        # Top left
    coord[7] = [0.005, 0.01]    # Top middle
    coord[8] = [0.01, 0.01]     # Top right

    dofs = np.zeros((9, 1), dtype=int)
    dofs[0] = [1]
    dofs[1] = [2]
    dofs[2] = [3]
    dofs[3] = [4]
    dofs[4] = [5]
    dofs[5] = [6]
    dofs[6] = [7]
    dofs[7] = [8]
    dofs[8] = [9]

    edof = np.zeros((8, 3), dtype=int)
    edof[0] = [1, 2, 5]
    edof[1] = [2, 3, 5]
    edof[2] = [3, 6, 5]
    edof[3] = [6, 9, 5]
    edof[4] = [9, 8, 5]
    edof[5] = [8, 7, 5]
    edof[6] = [7, 4, 5]
    edof[7] = [4, 1, 5]

    #element_markers = np.zeros((8), dtype=int)
    element_markers = [MARKER_CuCr] * 8

    bdofs = {
        MARKER_TCONST: [3, 6, 9],
        MARKER_BellOutside: [8, 9],
    }


    return (coord, edof, dofs, bdofs, element_markers)

def MakeThermMesh(geom) :
    mesh = cfm.GmshMeshGenerator(geom, mesh_dir=mesh_dir)

    dofs_pn = 1 # temp
    mesh.el_size_factor = el_sizef
    mesh.el_type = el_type
    mesh.dofs_per_node = dofs_pn

    coord, edof, dofs, bdofs, element_markers = mesh.create()

    return (coord, edof, dofs, bdofs, element_markers)

def AssembleThermStiffness(coord, edof, dofs, bdofs, element_markers, some_constants) :
    epCu = [some_constants["thickness"]]
    epTi = [some_constants["thickness"]]
    n_dofs = np.size(dofs)
    ex, ey = cfc.coordxtr(edof, coord, dofs)
    DCu = np.array([[some_constants["KCu"], 0.0], [0.0, some_constants["KCu"]]])  # Thermal transfer constitutive matrix
    DTi = np.array([[KTi, 0.0], [0.0, KTi]])  # Thermal transfer constitutive matrix
    K = np.zeros([n_dofs, n_dofs])

    for i in range(len(edof)):
        if element_markers[i] == MARKER_TiAlloy:
            Ke = cfc.flw2te(ex[i], ey[i], epTi, DTi)
        else:
            Ke = cfc.flw2te(ex[i], ey[i], epCu, DCu)
        K = cfc.assem(edof[i], K, Ke)

    return K

def MakeThermBC(F, bdofs, edof, coord, some_constants) :
    edges = nodesToEdges(bdofs, edof)

    bc, bc_value = np.array([], 'i'), np.array([], 'f')
    if MARKER_TCONST in bdofs.keys():
        bc, bc_value = cfu.applybc(bdofs, bc, bc_value, MARKER_TCONST, TCONST, 0)

    
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
        F[e[0]-1] += l*some_constants["thickness"]*ChamberHeating/2
        F[e[1]-1] += l*some_constants["thickness"]*ChamberHeating/2

    # Add cooling BC to outside of nozzle
    for e in edges[MARKER_ChamberOutside] :
        x1, y1 = coord[e[0]-1]
        x2, y2 = coord[e[1]-1]
        dx = x1-x2
        dy = y1-y2
        l = np.sqrt(dx*dx + dy*dy)
        F[e[0]-1] -= l*some_constants["thickness"]*ChamberCooling/2
        F[e[1]-1] -= l*some_constants["thickness"]*ChamberCooling/2

    # Add convecton BC
    KModifier = np.zeros([len(F), len(F)])
    for e in edges[MARKER_BellOutside] :
        #print("e: ", e)
        x1, y1 = coord[e[0]-1]
        x2, y2 = coord[e[1]-1]
        dx = x1-x2
        dy = y1-y2
        #print("dx, dy: ", dx, dy)
        l = np.sqrt(dx*dx + dy*dy)
        #print("l: ", l)
        #print(l, some_constants["thickness"], some_constants["AlphaConvection"], some_constants["Tinfty"])
        #print("df: ", l*some_constants["thickness"]*some_constants["AlphaConvection"]*some_constants["Tinfty"]/2)
        F[e[0]-1] += l*some_constants["thickness"]*some_constants["AlphaConvection"]*some_constants["Tinfty"]/2 #0.005 * 0.010 * 1000 * 20 / 2 = 5*1*0.01*10 =
        F[e[1]-1] += l*some_constants["thickness"]*some_constants["AlphaConvection"]*some_constants["Tinfty"]/2
        KModifier[e[0]-1][e[0]-1] += l * some_constants["thickness"] * some_constants["AlphaConvection"] /3
        KModifier[e[1]-1][e[1]-1] += l * some_constants["thickness"] * some_constants["AlphaConvection"] /3
        KModifier[e[0]-1][e[1]-1] += l * some_constants["thickness"] * some_constants["AlphaConvection"] /6
        KModifier[e[1]-1][e[0]-1] += l * some_constants["thickness"] * some_constants["AlphaConvection"] /6

    #print(F)
    #print(KModifier)

    return F, bc, bc_value, KModifier

def plotTherm(a, coord, edof) :
    UNIT = 1 / 2.54
    wcm, hcm = 35, 10

    fmt = '%1.2f'
    fig, ax = plt.subplots(figsize=(wcm * UNIT, hcm * UNIT))

    v = np.asarray(a)
    x, y = coord.T
    edof_tri = cfv.topo_to_tri(edof)
    tri = plt.tripcolor(x, y, edof_tri - 1, v.ravel(), shading="gouraud")

    fig.colorbar(tri, ax=ax, label='Temperature [K]', format=fmt)
    plt.title("Temperature distribution at equilibrium")

    plt.show()

def MakeCapacityMatrix(dofs, element_markers, some_constants) -> np.array :
    C = np.zeros((len(dofs), len(dofs)))
    for e in dofs:      # I have been a bad boy och itererat över dofsen istället för över elementen!!
                        # Detta bör dock vara fixat i MakeCapacityMatrix2()
        for i in e :
            if element_markers[i] == MARKER_CuCr :
                C[i-1, i-1] = some_constants["CCu"]
            if element_markers[i] == MARKER_TiAlloy :
                C[i-1, i-1] = CTi
    return C

def MakeCapacityMatrix2(dofs, edof, element_markers, some_constants) :
    C = np.zeros((len(dofs), len(dofs)))

    for i in range(len(edof)) :
        if element_markers[i] == MARKER_CuCr :
            for d in edof[i]:
                C[d-1, d-1] = some_constants["CCu"]
        elif element_markers[i] == MARKER_TiAlloy :
            for d in edof[i]:
                C[d-1, d-1] = CTi
    return C


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

def MakeMechTestMesh() :
    coord = np.zeros((9, 2))
    coord[0] = [-0.005, -0.005]    # Bottom left
    coord[1] = [0, -0.005]         # Bottom middle
    coord[2] = [0.005, -0.005]     # Bottom right
    coord[3] = [-0.005, 0]         # Middle left
    coord[4] = [0, 0]              # Center
    coord[5] = [0.005, 0]          # Middle right
    coord[6] = [-0.005, 0.005]     # Top left
    coord[7] = [0, 0.005]          # Top middle
    coord[8] = [0.005, 0.005]      # Top right

    dofs = np.zeros((9, 2), dtype=int)
    dofs[0] = [1, 2]
    dofs[1] = [3, 4]
    dofs[2] = [5, 6]
    dofs[3] = [7, 8]
    dofs[4] = [9, 10]
    dofs[5] = [11, 12]
    dofs[6] = [13, 14]
    dofs[7] = [15, 16]
    dofs[8] = [17, 18]


    edof = np.zeros((8, 6), dtype=int)
    edof[0] = [1, 2, 3, 4, 9, 10]
    edof[1] = [3, 4, 5, 6, 9, 10]
    edof[2] = [1, 2, 9, 10, 7, 8]
    edof[3] = [5, 6, 11, 12, 9, 10]
    edof[4] = [7, 8, 9, 10, 13, 14]
    edof[5] = [13, 14, 9, 10, 15, 16]
    edof[6] = [15, 16, 9, 10, 17, 18]
    edof[7] = [11, 12, 17, 18, 9, 10]

    #element_markers = np.zeros((8), dtype=int)
    element_markers = [0] * 8
    element_markers[0] = MARKER_CuCr
    element_markers[1] = MARKER_CuCr
    element_markers[2] = MARKER_CuCr
    element_markers[3] = MARKER_CuCr
    element_markers[4] = MARKER_CuCr
    element_markers[5] = MARKER_CuCr
    element_markers[6] = MARKER_CuCr
    element_markers[7] = MARKER_CuCr


    bdofs = {
        MARKER_ChamberOutside: [1, 2, 7, 8, 13, 14],
        MARKER_Inside: [5, 6, 11, 12, 17, 18],
        MARKER_TCONST: [15, 16, 17, 18]
    }


    return (coord, edof, dofs, bdofs, element_markers)

def AssembleMechStiffness(coord, edof, dofs, bdofs, element_markers, some_constants) :
    # Assemble plane strain matrix
    ptype = 1
    ep = np.array([ptype, some_constants["thickness"]])
    n_dofs = np.size(dofs)
    ex, ey = cfc.coordxtr(edof, coord, dofs)
    DCu = cfc.hooke(ptype, some_constants["ECu"], some_constants["VCu"])
    DTi = cfc.hooke(ptype, ETi, VTi)
    K = np.zeros([n_dofs, n_dofs])
    for i in range(len(edof)):
        if element_markers[i] == 1:
            Ke = cfc.plante(ex[i], ey[i], ep, DTi)
        else:
            Ke = cfc.plante(ex[i], ey[i], ep, DCu)
        K = cfc.assem(edof[i], K, Ke)

    return K, DCu, DTi, ex, ey, ep

def MakeMechBC(F, coord, dofs, bdofs, edof, some_constants) :
    bnods = {key: np.divide(bdofs[key][1::2], 2) for key in bdofs}
    enods = edof[:,1::2]/2
    edges = nodesToEdges(bnods, enods)                                               # Nodes to edges is knas. Använder noder istället för dofs

    bc, bc_value = np.array([], 'i'), np.array([], 'f')
    if MARKER_ChamberOutside in bdofs.keys():
        bc, bc_value = cfu.applybc(bdofs, bc, bc_value, MARKER_ChamberOutside, 0.0)
    if MARKER_QN_0 in bdofs.keys():
        bc, bc_value = cfu.applybc(bdofs, bc, bc_value, MARKER_QN_0, 0.0, 2)
    if MARKER_TCONST in bdofs.keys():
        cfu.applyforce(bdofs, F, MARKER_TCONST, -150e6*some_constants["thickness"]*0.005/2, 2) # pressure = 100 MPa, thickness = 0.01, sidelenght = 0.01 => 10 KN


    # Calculate force from pressure inside chamber
    for e in edges[MARKER_Inside]:
        #print(e)
        x1, y1 = coord[int(e[0] - 1)]
        #print("x1, y1: ", x1, y1)
        x2, y2 = coord[int(e[1] - 1)]
        #print("x2, y2: ", x2, y2)
        dx = x1 - x2
        dy = y1 - y2
        #print("d: ", dx, dy)
        fx = -some_constants["thickness"]*some_constants["InsidePressure"]*dy
        fy = some_constants["thickness"]*some_constants["InsidePressure"]*dx

        wrong_way = False
        for t in enods :
            if e[0] in t and e[1] in t :
                for i in t :
                    if i != e[0] and i != e[1] :
                        inside = i
                        x3, y3 = coord[int(i-1)]
                        dx = x3 - x1
                        dy = y3 - y1
                        if fx*dx + fy*dy < 0 :
                            wrong_way = True

        if wrong_way :
            fx *= -1
            fy *= -1
        #print("f: ", fx, fy)
        F[dofs[int((e[0]) - 1)][0] - 1] += fx / 2
        F[dofs[int((e[0]) - 1)][1] - 1] += fy / 2

        F[dofs[int(e[1] - 1)][0] - 1] += fx / 2
        F[dofs[int(e[1] - 1)][1] - 1] += fy / 2
    print(F)
    print(bc, bc_value)

    return F, bc, bc_value



if __name__=="__main__":
    #test(plot=True)

    temps = statTherm(True)

    #dynTherm(True)

    #Mech(plot=True, temps=temps)