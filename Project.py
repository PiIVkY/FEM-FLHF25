import calfem.geometry as cfg
import calfem.mesh as cfm

import calfem.core as cfc
import calfem.utils as cfu

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use('TkAgg')

import calfem.vis_mpl as cfv

import numpy as np

mesh_dir = "./"
MARKER_CuCr = 0
MARKER_TiAlloy = 1
MARKER_Inside = 2
MARKER_BellOutside = 3
MARKER_ChamberOutside = 4

def MakeGeometry() :
    g = cfg.geometry()
    g.point([0, 0], 0)
    g.point([0, 0.5], 1)
    g.point([1.8, 0.5], 2) # x-koordinaten är fel, ändra till exakt värde
    g.point([3.5, 1.1], 3) # top of ellipse
    g.point([3.5, 0.9], 4) # inside of top of ellipse
    g.point([2, 0.2], 5) # x-koordinaten är fel, ändra till exakt värde
    g.point([3.5, 0], 6) # center of ellipse
    g.point([1.4+0.2, 0.2], 7) # x-koordinaten är fel, ändra till exakt värde
    g.point([1.4, 0.3], 8)
    g.point([0.7, 0.3], 9)
    g.point([0.7, 0], 10)
    g.point([1.4, 0], 11)


    g.spline([0, 1])
    g.spline([1, 2])
    g.ellipse([2, 6, 0, 3])   #[startpoint, centerpoint, mAxisPoint, endpoint]
    g.spline([3, 4])
    g.ellipse([5, 6, 0, 4])
    g.spline([5, 7])
    g.circle([7, 11, 8])
    g.spline([8, 9])
    g.spline([9, 10])
    g.spline([10, 0])
    mesh = cfm.GmshMeshGenerator(g, mesh_dir=mesh_dir)

    fig, ax = plt.subplots()
    cfv.draw_geometry(
    g,
    label_curves=True,
    title="Geometry: Computer Lab Exercise 2"
    )
    plt.show()





if __name__=="__main__":
    MakeGeometry()
    SteadyStateSim()
    SteadyStatePlot()
    TransientSim()
    TransientPlot()
    VonMisesSim()
    VonMisesPlot()