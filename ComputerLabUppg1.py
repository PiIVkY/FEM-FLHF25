import calfem.geometry as cfg
import calfem.mesh as cfm

import calfem.core as cfc
import calfem.utils as cfu

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use('TkAgg')

import calfem.vis_mpl as cfv

import numpy as np

num_elements = 3
dof = 1

x_0 = 2
x_1 = 8

ex = np.zeros((num_elements+1, 2))
ey = np.zeros((num_elements+1, 2))

for i in range(num_elements+1):
    ex[i] = [x_0 + (x_1-x_0)*i/(num_elements+1), x_0 + (x_1-x_0)*(i+1)/(num_elements+1)]
    ey[i] = [0, 0]
print(ex)

edof = np.zeros((num_elements, 4))

for i in range(num_elements):
    edof[i] = [2*i, 2*i+1, 2*i+2, 2*i+3]
print(edof)

K = np.zeros((num_elements+1, num_elements+1))
kei = np.zeros((4,4))

E = 1
A = np.array ([1e-4, 1e-4, 1e-4])

for i in range(num_elements):
    kei = cfc.bar2e(ex[i, :], ey[i, :], [E, A[i]])
    cfc.assem(edof[i, :], K, kei)

F = np.zeros((num_elements, 1))
F[-2] = 15

bc_dof = np.array([1])
bc_val = np.array([0])

a, r = cfc.solveq(K, F, bc_dof , bc_val)

print("Temp:")
print(a)
print("Värme:")
print(r)