import calfem.geometry as cfg
import calfem.mesh as cfm

import calfem.core as cfc
import calfem.utils as cfu

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use('TkAgg')

import calfem.vis_mpl as cfv

import numpy as np

num_elements = 20

x_0 = 2
x_1 = 8
Q=100

ex = np.zeros((num_elements, 2))

for i in range(num_elements):
    ex[i] = [x_0 + (x_1-x_0)*i/(num_elements), x_0 + (x_1-x_0)*(i+1)/(num_elements)]

edof = np.zeros((num_elements, 4))

for i in range(num_elements):
    edof[i] = [2*i, 2*i+1, 2*i+2, 2*i+3]
print(edof)

K = np.zeros((num_elements+1, num_elements+1))
kei = np.zeros((4,4))

E = 1
A = np.array ([1e-4, 1e-4, 1e-4])

for i in range(num_elements):
    cfc.assem(edof[i, :], K, kei)

Fb = np.zeros((num_elements+1, 1))
Fb[num_elements] = -15*A

bc_dof = np.array([1])
bc_val = np.array([0])

Fl = np.zeros((num_elements+1, 1))
for i in range(len(Fl)):
    Fl[i] = Q*(x_1-x_0)/num_elements
Fl[0] = Q*(x_1-x_0)/(2*num_elements)
Fl[-1] = Q*(x_1-x_0)/(2*num_elements)

F = Fb + Fl

#print(K)
#print(F)
print(ex)
a, r = cfc.solveq(K, F, bc_dof , bc_val)

print("Temp:")
print(a)
print("Värme:")
print(r)