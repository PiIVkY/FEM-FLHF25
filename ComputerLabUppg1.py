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

x_0 = 2
x_1 = 8

ex = np.zeros((num_elements+1, 2))

for i in range(num_elements+1):
    ex[i] = [x_0 + (x_1-x_0)*i/(num_elements+1), x_0 + (x_1-x_0)*(i+1)/(num_elements+1)]

edof = np.zeros((num_elements,2), dtype=int)

for i in range(num_elements):
    edof[i] = [i+1, i+2]


K = np.zeros((num_elements+1, num_elements+1))
kei = np.zeros((4,4))

E = 5
A = 10

for i in range(num_elements):
    print(ex[i, :])
    print([E, A])
    kei = cfc.bar1e(ex[i, :], [E, A])
    cfc.assem(edof[i, :], K, kei)

F = np.zeros((num_elements+1, 1))
F[-2] = 15
print(ex)
bc_dof = np.array([1])
bc_val = np.array([0])

print(K)
a, r = cfc.solveq(K, F, bc_dof , bc_val)

print("Temp:")
print(a)
print("Värme:")
print(r)