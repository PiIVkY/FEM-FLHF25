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
el_sizef, el_type, dofs_pn = 0.08, 2, 3 # el_size = 0.08 ser nice ut, el_type vet ej vad den gör, dofs_pn: temp, skutning x, skjutning y
mesh_dir = "./"

MARKER_CuCr = 0
MARKER_TiAlloy = 1

MARKER_Inside = 2
MARKER_BellOutside = 3
MARKER_ChamberOutside = 4
MARKER_QN_0 = 5
MARKER_Material_Transition = 6

def MakeGeometry() :
    g = cfg.geometry()
    g.point([0, 0], 0)
    g.point([0, 0.5], 1)
    g.point([1.4, 0.5], 12) # material changes here
    g.point([3.5 - 1.7 * np.cos(np.arcsin(0.5/1.1)), 0.5], 2)
    g.point([3.5, 1.1], 3) # top of ellipse
    g.point([3.5, 0.9], 4) # inside of top of ellipse
    g.point([3.5 - 1.5 * np.cos(np.arcsin(0.2/0.9)), 0.2], 5)
    g.point([3.5, 0], 6) # center of ellipse
    g.point([1.4+np.sqrt(0.3*0.3 - 0.2*0.2), 0.2], 7)
    g.point([1.4, 0.3], 8)
    g.point([0.7, 0.3], 9)
    g.point([0.7, 0], 10)
    g.point([1.4, 0], 11)

    #1.7 cos(v) x + 1.1 sin(v) y, v:y=0.2 => sin(v) = 0.2/1.1 => v = arcsin(0.2/1.1) => x = 1.7 * cos(arcsin(0.2/1.1))

    g.spline([0, 1], 0, marker=MARKER_ChamberOutside)
    g.spline([1, 12], 1, marker=MARKER_ChamberOutside)
    g.spline([12, 2], 10, marker=MARKER_BellOutside)
    g.ellipse([2, 6, 0, 3], 2, marker=MARKER_BellOutside)   #[startpoint, centerpoint, mAxisPoint, endpoint]
    g.spline([3, 4], 3, marker=MARKER_BellOutside)
    g.ellipse([5, 6, 0, 4], 4, marker=MARKER_Inside)
    g.spline([5, 7], 5, marker=MARKER_Inside)
    g.circle([7, 11, 8], 6, marker=MARKER_Inside)
    g.spline([8, 9], 7, marker=MARKER_Inside)
    g.spline([9, 10], 8, marker=MARKER_Inside)
    g.spline([10, 0], 9, marker=MARKER_QN_0)
    g.spline([12, 8], 11, marker=MARKER_Material_Transition)

    g.surface([0, 1, 11, 7, 8, 9])
    g.surface([10, 2, 3, 4, 5, 6, 11])
    mesh = cfm.GmshMeshGenerator(g, mesh_dir=mesh_dir)

    mesh.el_size_factor = el_sizef
    mesh.el_type = el_type
    mesh.dofs_per_node = dofs_pn

    coord, edof, dofs, bdofs, element_markers = mesh.create()

    # Plotta den generade meshen
    fig, ax = plt.subplots()
    cfv.draw_geometry(
        g,
        label_curves=True,
        title="Geometry: Computer Lab Exercise 2"
    )
    plt.show()
    cfv.draw_mesh(coords=coord, edof=edof, dofs_per_node=mesh.dofs_per_node, el_type=mesh.el_type, filled=True)
    plt.show()


    # att göra: Boundary conditions sedan är meshen färdig
    # Eventuellt behöver även bc för tid = 0 sättas





if __name__=="__main__":
    MakeGeometry()
    SteadyStateSim()
    SteadyStatePlot()
    TransientSim()
    TransientPlot()
    VonMisesSim()
    VonMisesPlot()