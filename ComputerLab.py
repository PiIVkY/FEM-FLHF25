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
el_sizef, el_type, dofs_pn = 0.5, 2, 1
mesh_dir = "./"

# boundary markers
MARKER_T_1000=0
MARKER_T_100=1
MARKER_QN_0=2

def generate_mesh(show_geometry: bool):
    # initialize mesh
    g = cfg.geometry()
    # define parameters
    R = 2
    L = 10
    # add points
    g.point([0, 0], 0)
    g.point([R, 0], 1)
    g.point([L, 0], 2)
    g.point([L, L], 3)
    g.point([0, L], 4)
    g.point([0, R], 5)
    # define lines / circle segments
    g.circle([1, 0, 5], 0, marker=MARKER_T_1000)
    g.spline([1, 2], 1, marker=MARKER_QN_0)
    g.spline([2, 3], 2, marker=MARKER_T_100)
    g.spline([3, 4], 3, marker=MARKER_T_100)
    g.spline([4, 5], 4, marker=MARKER_QN_0)
    # define surface
    g.surface([0, 1, 2, 3, 4])
    # generate mesh
    mesh = cfm.GmshMeshGenerator(g, mesh_dir=mesh_dir)
    mesh.el_size_factor = el_sizef
    mesh.el_type = el_type
    mesh.dofs_per_node = dofs_pn
    coord, edof, dofs, bdofs, element_markers = mesh.create()
    # display mesh
    if show_geometry:
        fig, ax = plt.subplots()
        cfv.draw_geometry(
            g,
            label_curves=True,
            title="Geometry: Computer Lab Exercise 2"
        )
        plt.show()
        cfv.draw_mesh(coords=coord, edof=edof, dofs_per_node=mesh.dofs_per_node, el_type=mesh.el_type, filled=True)
        plt.show()
    # Boundary Conditions
    bc, bc_value = np.array([], 'i'), np.array([], 'f')
    bc, bc_value = cfu.applybc(bdofs, bc, bc_value, MARKER_T_1000, 1000, 1)
    bc, bc_value = cfu.applybc(bdofs, bc, bc_value, MARKER_T_100, 100, 1)
    # return
    return (coord, edof, dofs, bdofs, bc, bc_value, element_markers)

if __name__=="__main__":
    coord, edof, dofs, bdofs, bc, bc_value, element_marker = generate_mesh(show_geometry=False)
    Ex, Ey = cfc.coordxtr(edof, coord, dofs)

    ep = np.array([1])
    D = np.array([[1,0],[0,1]])
    K = np.zeros((dofs[-1][0], dofs[-1][0]))
    F = np.zeros((dofs[-1][0], 1))

    for i in range(len(edof)):
        Ke = cfc.flw2te(Ex[i], Ey[i], ep, D)
        K = cfc.assem(edof[i], K, Ke)

    a, r = cfc.solveq(K, F, bc, bc_value)

    Ed = cfc.extractEldisp(edof, a)
    Es = np.zeros((dofs[-1][0], 2))
    for i in range(500):
        Es[i], Et = cfc.flw2ts(Ex[i], Ey[i], D, Ed[i])

    #cfu.disp_h2("Ex")
    #cfu.disp_array(Ex, headers=["x0", "x1", "x2", "x3"])
    #cfu.disp_h2("Ey")
    #cfu.disp_array(Ey, headers=["x0", "x1", "x2", "x3"])
    #cfu.disp_h2("a")
    #cfu.disp_array(a)
    #cfu.disp_h2("Ed")
    #cfu.disp_array(Ed, headers=["ed0", "ed1", "ed2", "ed3"])

    cfv.eldraw2(Ex, Ey, [1, 2, 1], range(1, Ex.shape[0] + 1))
    cfv.eliso2_mpl(Ex, Ey, Ed)
    cfv.show_and_wait()