# imports
import numpy as np
import calfem.core as cfc
import matplotlib.pyplot as plt

import calfem.utils as cfu
import matplotlib as mpl

mpl.use('TkAgg')

import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis_mpl as cfv


# cute little function that does cute stuff like calculate x val from y val in an ellipse
# The ellipse is on the form (x/a)^2 + (y/b)^2 = 1
def x_coord_on_ellipse(a, b, y_val):
    return a * np.sqrt(1 - (y_val / b) ** 2)


# Boundary markers
MARKER_Q_IN = 1  # Boundary with Q_in
MARKER_Q_OUT = 2  # Boundary with Q_out
MARKER_SYMMETRY_AXIS = 3  # Boundary along symmetry axis
MARKER_ALPHA_C = 4  # Convection boundary
MARKER_CONNECTING_BOUNDARY = 5  # Boundary between elements

# Surface markers
MARKER_SURF_TI = 10
MARKER_SURF_CUCR = 20


# Geometry
def createGeometry():
    g = cfg.geometry()

    # Calculate the values for the x coordinates for 'difficult' points
    # along the rocket nozzle
    point_4_x_val = np.round(1.4 + x_coord_on_ellipse(0.3, 0.3, 0.2), 4)
    point_6_x_val = np.round(3.5 - x_coord_on_ellipse(1.5, 0.9, 0.2), 4)
    point_10_x_val = np.round(3.5 - x_coord_on_ellipse(1.7, 1.1, 0.5), 4)

    # Define the points of the geometry
    g.point([0, 0], 0)
    g.point([0.7, 0], 1)
    g.point([0.7, 0.3], 2)

    g.point([1.4, 0.3], 3)  # Lower point of boundary between Ti and CuCr

    g.point([point_4_x_val, 0.2], 4)
    g.point([1.4, 0], 5)  # Circle centre point
    g.point([point_6_x_val, 0.2], 6)
    g.point([3.5, 0], 7)  # Ellipse centre point
    g.point([3.5, 0.9], 8)
    g.point([3.5, 1.1], 9)
    g.point([point_10_x_val, 0.5], 10)

    g.point([1.4, 0.5], 11)  # Upper point of boundary between Ti and CuCr

    g.point([0, 0.5], 12)

    # Surface 1, Ti
    g.spline([0, 1], 0, marker=MARKER_SYMMETRY_AXIS)  # 'Edge' along symmetry axis
    g.spline([1, 2], 1, marker=MARKER_Q_IN)
    g.spline([2, 3], 2, marker=MARKER_Q_IN)
    g.spline([3, 11], 3, marker=MARKER_CONNECTING_BOUNDARY)  # Boundary between the surfaces, reused for CuCr surface
    g.spline([11, 12], 4, marker=MARKER_Q_OUT)
    g.spline([12, 0], 5, marker=MARKER_Q_OUT)

    g.surface([0, 1, 2, 3, 4, 5], marker=MARKER_SURF_TI)

    # Surface 2, CuCr
    g.circle([3, 5, 4], 6, marker=MARKER_Q_IN)
    g.spline([4, 6], 7, marker=MARKER_Q_IN)
    g.ellipse([6, 7, 0, 8], 8, marker=MARKER_Q_IN)
    g.spline([8, 9], 9, marker=MARKER_ALPHA_C)
    g.ellipse([9, 7, 0, 10], 10, marker=MARKER_ALPHA_C)
    g.spline([10, 11], 11, marker=MARKER_ALPHA_C)

    g.surface([3, 6, 7, 8, 9, 10, 11], marker=MARKER_SURF_CUCR)

    return g


def nodesToEdges(nodes: dict, enod: np.array) -> dict:
    """ Returns a list of edges given nodes

    Args :
    nodes (dict): nodes on boundary
    enod (np.array): element connectivity matrix

    Returns :
    dict: dict of edges on boundary, with markers as keys
    """
    # Initialize edges dict
    edges = {}
    for key in nodes.keys():
        edges[key] = []

    for con in zip(enod):
        for key in nodes.keys():
            I = np.intersect1d(con, nodes[key])
            if len(I) == 2:
                edges[key].append(I)
    return edges


# Function to apply line flux along a given boundary
def apply_boundary_flux(f, edges_on_boundary, coords, flux):
    for edge in edges_on_boundary:
        n1, n2 = edge
        n1_coords, n2_coords = coords[n1 - 1], coords[n2 - 1]
        delta_l = np.linalg.norm(n1_coords - n2_coords)
        flux_per_node = flux * delta_l / 2
        f[n1 - 1] -= flux_per_node
        f[n2 - 1] -= flux_per_node


# Function to determine element load vector and element stiffness matrix for convection
def Newton_boundary_assem(K, f, edges_on_convection_boundary: list, coords, T_inf, alpha_c):
    for edge in edges_on_convection_boundary:
        n1, n2 = edge
        n1_coords, n2_coords = coords[n1 - 1], coords[n2 - 1]
        delta_l = np.linalg.norm(n1_coords - n2_coords)

        K_ce = t * alpha_c * delta_l / 6 * np.array([[2, 1], [1, 2]])
        f_ce = t * T_inf * alpha_c * delta_l / 2 * np.ones((2, 1))

        # Assemble into load vector and global stiffness matrix
        cfc.assem(np.array([n1, n2]), K, K_ce, f, f_ce)


g = createGeometry()
mesh = cfm.GmshMesh(g)

mesh.el_type = 2  # Triangular elements
mesh.dofs_per_node = 1  # Degrees of freedom per node
mesh.el_size_factor = 0.04  # Element size factor

# Create mesh
coords, edof, dofs, bdofs, elementmarkers = mesh.create()

# Plot mesh
cfv.figure()
cfv.draw_mesh(
    coords=coords,
    edof=edof,
    dofs_per_node=mesh.dofs_per_node,
    el_type=mesh.el_type,
    filled=True,
    title="Mesh for rocket nozzle"
)
plt.show(block=False)

##################################
# Thermal conductivity k
k_Ti = 12  # W/mK
k_CuCr = 323  # W/mK

# Heat flux
q_in = -5000  # W/m^2
q_out = 1000  # W/m^2

# Convection constants
alpha_c = 50  # W/m^2K
T_inf = 293  # K

# Starting temperature
T_0 = 293  # K

# Element thickness
t = 1.0  # m

n_dofs = dofs.size  # Number of nodes, since 1 dof per node
nelm = len(edof)  # Number of elements

# Empty global stiffness matrix
K = np.zeros((n_dofs, n_dofs))

ex, ey = cfc.coordxtr(edof, coords, dofs)

D_TI = np.array([[k_Ti, 0], [0, k_Ti]])
D_CUCR = np.array([[k_CuCr, 0], [0, k_CuCr]])
D_dict = {MARKER_SURF_TI: D_TI, MARKER_SURF_CUCR: D_CUCR}

ep = np.array([t])

for i in range(nelm):
    # Calculate the element matrix Ke
    element_D = D_dict[elementmarkers[i]]
    Ke = cfc.flw2te(ex[i], ey[i], ep, element_D)

    # Assemble Ke into the global stiffness matrix K
    cfc.assem(edof[i], K, Ke)

# Load vector
f = np.zeros((n_dofs, 1))

edges_on_boundary = nodesToEdges(bdofs, edof)

apply_boundary_flux(f, edges_on_boundary[MARKER_Q_IN], coords, q_in)
apply_boundary_flux(f, edges_on_boundary[MARKER_Q_OUT], coords, q_out)

Newton_boundary_assem(K, f, edges_on_boundary[MARKER_ALPHA_C], coords, T_inf, alpha_c)

# Solving the system
bcPrescr, bcVal = np.array([], 'i'), np.array([], 'f')
a, Q = cfc.solveq(K, f, bcPrescr, bcVal)

# Max and min temp fr fr no cap
print(f"max: {max(a)}")
print(f"min: {min(a)}")

# Nice plot, hopefully
cfv.figure()
cfv.draw_nodal_values_shaded(a, coords, edof, "stationary temperature distribution in nozzle")
cfv.colorbar()
plt.show(block=False)

#################################

# Density
DENSITY_TI = 4540  # kg/m^3
DENSITY_CUCR = 8890  # kg/m^3
density_dict = {MARKER_SURF_TI: DENSITY_TI, MARKER_SURF_CUCR: DENSITY_CUCR}

# Specific heat capacity, c_p
SPECIFIC_HEAT_TI = 590  # J/kgK
SPECIFIC_HEAT_CUCR = 377  # J/kgK
c_p_dict = {MARKER_SURF_TI: SPECIFIC_HEAT_TI, MARKER_SURF_CUCR: SPECIFIC_HEAT_CUCR}


def plantml(ex: np.array, ey: np.array, s: float):
    """
    Computes the integral of the form-functions over a 3-node triangle element
        Me = int(s*N^T*N)dA

    Inputs:
        ex:     element x-coordinates
        ey:     element y-coordinates
        s:      constant scalar, e.g. density*thickness

    Outputs:
        Me:     integrated element matrix
    """
    if not ex.shape == (3,) or not ey.shape == (3,):
        raise Exception("Incorrect shape of ex or ey: {0}, {1} but should be (3,)".format(ex.shape, ey.shape))

    # Compute element area
    Cmat = np.vstack((np.ones((3,)), ex, ey))
    A = np.linalg.det(Cmat) / 2

    # Set up quadrature
    g1 = [0.5, 0.0, 0.5]
    g2 = [0.5, 0.5, 0.0]
    g3 = [0.0, 0.5, 0.5]
    w = (1 / 3)

    # Perform numerical integration
    Me = np.zeros((3, 3))
    for i in range(0, 3):
        Me += w * np.array([
            [g1[i] ** 2, g1[i] * g2[i], g1[i] * g3[i]],
            [g2[i] * g1[i], g2[i] ** 2, g2[i] * g3[i]],
            [g3[i] * g1[i], g3[i] * g2[i], g3[i] ** 2]])

    Me *= A * s
    return Me


# Time stepping constants
number_of_steps = 100
t_final = 3600  # s = 1 h
h = t_final / number_of_steps  # time step
a_0 = np.full(a.shape, T_0)  # values at t=0, all 293 K

# Temperature values a at time step n are given by a_n
a_n = a_0

# Keeps track of the max and min temperature at every step
max_temp = np.zeros(number_of_steps + 1)
min_temp = np.zeros(number_of_steps + 1)
max_temp[0], min_temp[0] = T_0, T_0

# Plots the temperature at time t=0s
cfv.figure()
cfv.draw_nodal_values_shaded(a_n, coords, edof, "temperature at time t = 0 s")
cfv.colorbar()
plt.show(block=False)

# Computes the time stepping
for step in range(1, number_of_steps + 1):
    C = np.zeros((n_dofs, n_dofs))

    for i in range(nelm):
        # Calculate the element matrix Ce
        element_density = density_dict[elementmarkers[i]]
        element_c_p = c_p_dict[elementmarkers[i]]
        Ce = plantml(ex[i], ey[i], t * element_density * element_c_p)

        # Assemble Ce into C
        cfc.assem(edof[i], C, Ce)

    inv_C_K_h = np.linalg.inv(C + K * h)
    a_n = inv_C_K_h @ (f * h + C @ a_n)

    max_temp[step], min_temp[step] = max(a_n.flatten()), min(a_n.flatten())

    if step % 20 == 0:
        cfv.figure()
        cfv.draw_nodal_values_shaded(a_n, coords, edof,
                                     f"temperature distribution at time t = {step * h} s ({step * h / 60} min)")
        cfv.colorbar()
        plt.show(block=False)

# Plot max and min temperature over time
time_array = np.linspace(0, t_final / 60, number_of_steps + 1)
plt.figure()
plt.plot(time_array, max_temp, label='max temperature', color='orange')
plt.plot(time_array, min_temp, label='min temperature', color='blue')
plt.xlabel('Time [min]')
plt.ylabel('Temperature [K]')
plt.title('max/min temperature over time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show(block=False)

##################################

# Creating new markers, to clarify which bc applies to which edge
MARKER_U_0 = MARKER_Q_OUT
MARKER_T_0 = MARKER_ALPHA_C
MARKER_P_0 = MARKER_Q_IN

# Boundary condition values
p_0_bc = 1e6  # Pa
t_bc = 0.0
u_bc = 0.0

# Normal vectors
edges_along_P_0 = edges_on_boundary[MARKER_P_0]

# Coordinates for normal vectors
normal_vectors_x_coords = np.zeros(len(edges_along_P_0))
normal_vectors_y_coords = np.zeros(len(edges_along_P_0))
normal_vectors_x_directions = np.zeros(len(edges_along_P_0))
normal_vectors_y_directions = np.zeros(len(edges_along_P_0))
for i, edge in enumerate(edges_along_P_0):
    n1, n2 = edge
    first_node, last_node = -1, -1
    for el in edof:
        if n1 in el and n2 in el:
            index1 = np.where(el == n1)[0]
            index2 = np.where(el == n2)[0]

            if (index2 - index1) % 3 == 1:
                first_node = n2
                last_node = n1
            else:
                first_node = n1
                last_node = n2
            break  # When the element is found, the loop doesn't need to continue
    # Get the coords for the edge
    first_coords, last_coords = coords[first_node - 1], coords[last_node - 1]
    # Coordinates for normal vector (middle of edge)
    normal_vectors_x_coords[i], normal_vectors_y_coords[i] = 1 / 2 * (last_coords + first_coords)
    # Direction for normal vector
    dir_x, dir_y = last_coords - first_coords
    normal_vectors_x_directions[i], normal_vectors_y_directions[i] = np.array([-dir_y, dir_x])

cfv.figure()
cfv.draw_mesh(
    coords=coords,
    edof=edof,
    dofs_per_node=mesh.dofs_per_node,
    el_type=mesh.el_type,
    filled=True,
    title="Mesh for rocket nozzle")
plt.quiver(normal_vectors_x_coords, normal_vectors_y_coords,
           normal_vectors_x_directions, normal_vectors_y_directions, color="blue")

# Material parameters
# Young's modulus, E
E_TI = 108e9  # Pa
E_CUCR = 139e9  # Pa
E_dict = {MARKER_SURF_TI: E_TI, MARKER_SURF_CUCR: E_CUCR}

# Poisson's ratio, v (nu)
V_TI = 0.22
V_CUCR = 0.18
nu_dict = {MARKER_SURF_TI: V_TI, MARKER_SURF_CUCR: V_CUCR}

# Expansion coefficient, alpha
ALPHA_TI = 8.8e-6  # 1/K
ALPHA_CUCR = 17.0e-6  # 1/K
alpha_dict = {MARKER_SURF_TI: ALPHA_TI, MARKER_SURF_CUCR: ALPHA_CUCR}


# Defines a function to calculate the D-matrix for Ti and CuCr, and then calculates them
def D_matrix(E, v):
    D_mat = E / ((1 + v) * (1 - 2 * v)) * np.array([[1 - v, v, 0],
                                                    [v, 1 - v, 0],
                                                    [0, 0, 1 / 2 * (1 - 2 * v)]])
    return D_mat


D_MECHANICAL_TI = D_matrix(E_TI, V_TI)
D_MECHANICAL_CUCR = D_matrix(E_CUCR, V_CUCR)
D_mechanical_dict = {MARKER_SURF_TI: D_MECHANICAL_TI, MARKER_SURF_CUCR: D_MECHANICAL_CUCR}

n_dofs_mech = 2 * n_dofs


# Fixes the old edof matrix, to give space for 2 dof per node
def fix_edof_to_2_dof(edof):
    m, n = edof.shape
    new_edof = np.zeros((m, 2 * n), 'i')

    for i in range(m):
        for j in range(n):
            new_edof[i, 2 * j] = 2 * edof[i, j] - 1
            new_edof[i, 2 * j + 1] = 2 * edof[i, j]

    return new_edof


# Fixes the old edof matrix, to give space for 2 dof per node
def fix_bdof_to_2_dof(bdof: dict):
    new_bdof = {}
    for key in bdof.keys():
        new_bdof[key] = []

        for dof in bdof[key]:
            new_bdof[key].append(2 * dof - 1)
            new_bdof[key].append(2 * dof)

    return new_bdof


# Creates new edof, since dofs_per_node = 2 now
edof_mech = fix_edof_to_2_dof(edof)

# Create new bdof dict, since dofs_per_node = 2 now
bdofs_mech = fix_bdof_to_2_dof(bdofs)

# define ep necessary for cfc.plante(), with ptype = 2 (plane strain)
ep = np.array([2, t])

# New K-matrix, for mechanical problem
K_mechanical = np.zeros((n_dofs_mech, n_dofs_mech))

# The temperature distribution at the final time t=1h
a_final = a_n

# New global f for mechanical problem
f_mechanical = np.zeros((n_dofs_mech, 1))

# Assemble the element K-matrices into the global stiffness matrix for mechanical problem
# and fe into global f
# ex, ey from (a) are still applicable
for eltopo, nodes_in_el, elx, ely, elmarker in zip(edof_mech, edof, ex, ey, elementmarkers):
    el_D_matrix = D_mechanical_dict[elmarker]
    el_nu = nu_dict[elmarker]
    el_alpha = alpha_dict[elmarker]
    el_E = E_dict[elmarker]

    # Average out the nodal temperatures to get a mean temperature for the element,
    # since the integral is over the entire element
    meanT = 1 / 3 * (a_final[nodes_in_el[0] - 1] + a_final[nodes_in_el[1] - 1]
                     + a_final[nodes_in_el[2] - 1])[0]
    # Calculates the temperature difference from the
    deltaT = meanT - T_0

    # Element contribution due to thermal expansion
    epsilon_deltaT = el_alpha * deltaT * np.array([[1], [1], [0]])
    es = el_D_matrix @ epsilon_deltaT
    fe = cfc.plantf(elx, ely, ep, es.T)

    # el_D_matrix = cfc.hooke(2, el_E, el_nu)
    Ke = cfc.plante(elx, ely, ep, el_D_matrix)
    cfc.assem(eltopo, K_mechanical, Ke, f_mechanical, fe)

# Apply Dirichlet boundary condition, Ti part is fastened
bc, bcValues = np.array([], 'i'), np.array([], 'f')
bc, bcValues = cfu.apply_bc(bdofs_mech, bc, bcValues, MARKER_U_0, u_bc, 0)
bc, bcValues = cfu.apply_bc(bdofs_mech, bc, bcValues, MARKER_SYMMETRY_AXIS, 0, 2)

# Solve for delta_u from thermal
u_thermal, _ = cfc.solveq(K_mechanical, f_mechanical, bc, bcValues)
cfv.figure()
cfv.draw_displacements(u_thermal, coords, edof_mech, 2, 2, draw_undisplaced_mesh=True, magnfac=100,
                       title="Displacement due to thermal expansion,\n with a magnification factor of 100")
plt.xlabel("length, [m]")
plt.ylabel("length, [m]")
plt.show(block=False)

# Force applied by the pressure p_0 inside of the nozzle to each node
for i, edge in enumerate(edges_along_P_0):
    n1, n2 = edge
    deltaL = np.linalg.norm(coords[n1 - 1] - coords[n2 - 1])

    # Normal vector
    x_dir = normal_vectors_x_directions[i]
    y_dir = normal_vectors_y_directions[i]

    normal_vec_length = np.linalg.norm((x_dir, y_dir))

    normal_unit_vector = np.array([x_dir, y_dir]) / normal_vec_length

    # p_0 as force (p_0 * area)
    p_0_as_force = p_0_bc * normal_unit_vector * deltaL * t

    # Apply force to f_mechanical
    f_mechanical[2 * (n1 - 1)] -= p_0_as_force[0]
    f_mechanical[2 * (n1 - 1) + 1] -= p_0_as_force[1]

# Solve for the displacements due to pressure and thermal expansion
u, _ = cfc.solveq(K_mechanical, f_mechanical, bc, bcValues)

# Extract the element displacements from u
ed = cfc.extract_eldisp(edof_mech, u)

# Create a list for the von Mises stress in each element
von_Mises_stress_per_element = np.zeros(nelm)

# element properties, for clarification
ep = [2, t]  # t = 1.0 m

# Calculate von Mises stress in each element
for i in range(len(edof_mech)):
    elmarker = elementmarkers[i]
    el_D_matrix = D_mechanical_dict[elmarker]
    el_nu = nu_dict[elmarker]
    el_alpha = alpha_dict[elmarker]
    el_E = E_dict[elmarker]

    # Average out the nodal temperatures to get a mean temperature for the element,
    # since the integral is over the entire element
    meanT = 1 / 3 * (a_final[nodes_in_el[0] - 1] + a_final[nodes_in_el[1] - 1]
                     + a_final[nodes_in_el[2] - 1])[0]
    # Calculates the temperature difference from the
    deltaT = meanT - T_0

    es, _ = cfc.plants(ex[i], ey[i], ep, el_D_matrix, ed[i])

    epsilon_deltaT = el_alpha * deltaT * np.array([[1], [1], [0]])
    D_epsilon_deltaT = el_D_matrix @ epsilon_deltaT

    # Calculating different values needed for sigma_eff
    sigma_x = es[0][0] - D_epsilon_deltaT[0]
    sigma_y = es[0][1] - D_epsilon_deltaT[1]
    tau_xy = es[0][2] - D_epsilon_deltaT[2]
    sigma_z = el_nu * (sigma_x + sigma_y) - el_alpha * deltaT * el_E

    # Calculating sigma_eff, tau_xz = tau_yz = 0
    sigma_eff = np.sqrt(
        sigma_x ** 2 + sigma_y ** 2 + sigma_z ** 2 - sigma_x * sigma_y - sigma_x * sigma_z - sigma_y * sigma_z + 3 * tau_xy ** 2)

    von_Mises_stress_per_element[i] = sigma_eff


# Function to go from element von Mises stress to node
def elmToNode(eV: np.array, edof: np.array) -> np.array:
    """
    Estimates nodal values from element - based values

    Args :
    eV (np.array): element values
    edof (np.array): element connectivity matrix

    Returns :
    np. array : nodal - based values
    """

    nnod: int = np.max(edof)
    ne: int = 0
    nV = np.zeros((nnod,))
    # Loop over nodes
    for n in range(0, nnod):
        ne = 0
        # Check which elements contain the node
        for e, eldof in enumerate(edof):
            # If e contains the node add the elemental value
            if ((n + 1) in eldof):
                ne += 1
                nV[n] += eV[e]

        # Divide by total number of elements
        nV[n] /= ne
    return nV


nodal_values = elmToNode(von_Mises_stress_per_element, edof)

cfv.figure()
cfv.draw_nodal_values_shaded(nodal_values, coords, edof, f"von Mises stress distribution at t = 1 h")
cfv.colorbar()

print(f"maximum von Mises stress: {np.max(nodal_values) / 1e6} MPa")

plt.show()