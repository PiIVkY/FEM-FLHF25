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
el_sizef = 0.04 # Element size factor
el_type=  2 # Triangular elements

mesh_dir = "./"

# Surface markers
MARKER_CuCr = 0
MARKER_TiAlloy = 1

# Constants for the Ti-alloy
ETi = 108e9     # Young's modulus (Pa)
VTi = 0.22      # Poisson's ratio
AlphaTi = 8.8e-6# thermal expansion coefficient (1/K)
RhoTi = 4540    # Density (kg/m^3)
CTi = 590       # Specific heat (J/KgK)
KTi = 12        # Thermal conductivity (W/mK)

# Constants for the Cu-Cr-alloy
ECu = 139e9    # Young's modulus (Pa)
VCu = 0.18     # Poisson's ratio
AlphaCu = 17e-6# thermal expansion coefficient (1/K)
RhoCu = 8890   # Density (kg/m^3)
CCu = 377      # Specific heat (J/KgK)
KCu = 323      # Thermal conductivity (W/mK)

# Boundary markers and values for relevant boundary conditions
MARKER_Inside = 2
ChamberHeating = 5000 # (W/m^2)
p_0 = 1e6  # (Pa)

MARKER_BellOutside = 3

MARKER_ChamberOutside = 4
ChamberCooling = 1000 # (W/m^2)

MARKER_QN_0 = 5
MARKER_Material_Transition = 6 

# Other constants
AlphaConvection = 50  # (W/m^2K)
thickness = 1 # (m)
Tinfty = 293  # (K)


def statTherm(plot) :
    """
    Solves the stationary thermal problem for the rocket nozzle

    Inputs:
        plot: Toggles the plots on (True) and off (False)
    """

    # Creates the mesh for the thermal problem
    coord, edof, dofs, bdofs, element_markers = MakeThermMesh(NozzleGeom())

    # Assembles the K-matrix
    K = AssembleThermStiffness(coord, edof, dofs, element_markers)

    # Computes the boundary coditions and inserts them into the F-matrix
    # Also computes the change to K due to the convection boundary contition
    F, bc, bc_value, KModifier = MakeThermBC(coord, edof, dofs, bdofs)
    
    # Update K with changes from convection boundary contition
    K = K + KModifier  

    # Solve the stationary thermal problem
    a, r = cfc.solveq(K, F, bc, bc_value)

    # Finds and prints max temp, min temp and biggest temperature difference
    min = 100000
    max = 0
    diff = 0
    TDiff = 293
    
    for i in range(len(a)):
        if a[i] > max:
            max = a[i]
            maxcord = coord[i]
        if a[i] < min:
            min = a[i]
            mincord = coord[i]
        if a[i] - TDiff > diff:
            diff = a[i] - TDiff
            diffcord = coord[i]
    
    print("Stationär maxtemp:", max, " på plats ", maxcord, "Stationär mintemp:", min, " på plats ", mincord)
    print("Störst temperaturskillnad", diff, " på plats ", diffcord)

    if plot:  
        # Plots the nozzle geometry
        cfv.draw_geometry(
            NozzleGeom(),
            label_curves=True,
            title="Geometry of the rocket nozzle"
        )
        plt.xlabel("x-coordinate [m]")
        plt.ylabel("y-coordinate [m]")
        plt.show()

        # Plots the mesh
        cfv.draw_mesh(coords=coord, edof=edof, dofs_per_node=1, el_type=2, filled=True)
        plt.title("Mesh of the rocket nozzle")
        plt.xlabel("x-coordinate [m]")
        plt.ylabel("y-coordinate [m]")
        plt.show()

        # Plots the temperature distribution at equilibrium
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
        plt.xlabel("x-coordinate [m]")
        plt.ylabel("y-coordinate [m]")
        plt.show()

def dynTherm(plot) :
    """
    Solves the dynamic thermal problem for the rocket nozzle

    Inputs:
        plot: Toggles the plots on (True) and off (False)

    Returns the temperature distribution in the rocket nozzle after 1 hour
    """

    # Creates the mesh for the thermal problem
    coord, edof, dofs, bdofs, element_markers = MakeThermMesh(NozzleGeom())

    # Assembles the K-matrix
    K = AssembleThermStiffness(coord, edof, dofs, element_markers)

    # Computes the boundary coditions and inserts them into the F-matrix
    # Also computes the change to K due to the convection boundary contition
    F, bc, bc_value, KModifier = MakeThermBC(coord, edof, dofs, bdofs)
    
    # Update K with changes from convection boundary contition
    K = K + KModifier  

    # Computes the capacity matrix 
    C = MakeCapacityMatrix(coord, dofs, edof, element_markers)

    # Initializes array with the temperatures in the nozzle at t=0
    a0 = np.ones((len(dofs), 1)) * Tinfty

    # Values for the time stepping method
    dt, tottime, alpha = 10, 3600, 1
    smoothness = 0.5
    times = [100 * i * smoothness for i in range(int(tottime / (100 * smoothness)))]

    # Carries out the time stepping method
    modhist, dofhist = cfc.step1(K, C, F, a0, bc, [dt, tottime, alpha], times, dofs=np.array([]))

    if plot:
        # Finds and plots the max and min temperature in the nozzle over all time steps
        Tmax = np.empty(len(times))
        Tmin = np.empty(len(times))
        for i in range(len(times)):
            Tmax[i] = np.max(modhist['a'].transpose()[i])
            Tmin[i] = np.min(modhist['a'].transpose()[i])

        xArr = np.arange(0, tottime, int(tottime/len(times)))

        plt.plot(xArr, Tmax, label = "Maximum temperature", color = 'r')
        plt.plot(xArr, Tmin, label = "Minimum temperature", color = 'b')
        plt.xlabel("Time [s]")
        plt.ylabel("Temperature [K]")
        plt.legend()
        plt.title("Maximum and minimum temperature in the rocket nozzle for the first hour")
        plt.show()

        # Generates an animated plot for the temperatures at all time steps
        UNIT = 1 / 2.54
        wcm, hcm = 35, 10
        fig, (ax, cbax) = plt.subplots( 1, 2, width_ratios=[10, 1], figsize=(wcm * UNIT, hcm * UNIT))

        x, y = coord.T
        fmt = '%1.2f'
        v = np.asarray(modhist["a"].transpose()[0])
        edof_tri = cfv.topo_to_tri(edof)
        im = ax.tripcolor(x, y, edof_tri - 1, v.ravel(), shading="gouraud")
        fig.colorbar(im, ax=ax, cax = cbax, label='Temperature [K]', format=fmt)
        
        i0 = 0
        tx = ax.text(3, 0.1, str(i0))

        # Function which assists in the animation
        def animate(i) :
            v = np.asarray(modhist["a"].transpose()[i])
            im = ax.tripcolor(x, y, edof_tri - 1, v.ravel(), shading="gouraud")
            fig.colorbar(im, ax=ax, cax = cbax, label='Temperature [K]', format=fmt)
            tx.set_text("frame: " + str(i))

        plt.title("Temperature distribution at equilibrium")

        # Plots the animation
        ani = animation.FuncAnimation(fig=fig, func=animate, frames = len(times), interval=(smoothness*200))
        plt.show()

    return np.matrix([modhist["a"][:,-1]]).transpose()

def Mech(plot, temps) :
    """
    Solves the mechanical problem for the rocket nozzle and computes the von Mises stress

    Inputs:
        plot: Toggles the plots on (True) and off (False)
        temps: Temperatures in the rocket nozzle after one hour
    """

    # Generate the mesh for the mechanical problem with 2 degrees of freedom per node
    mesh = cfm.GmshMeshGenerator(NozzleGeom(), mesh_dir=mesh_dir)

    dofs_pn = 2 

    mesh.el_size_factor = el_sizef
    mesh.el_type = el_type
    mesh.dofs_per_node = dofs_pn

    coord, edof, dofs, bdofs, element_markers = mesh.create()

    # Finds edges of the new mesh
    bnods = {key: np.divide(bdofs[key][1::2], 2) for key in bdofs}
    enods = edof[:,1::2]/2
    edges = nodesToEdges(bnods, enods)

    # Coordinates for normal vectors
    normalVect_x_coords = np.zeros(len(edges[MARKER_Inside]))
    normalVect_y_coords = np.zeros(len(edges[MARKER_Inside]))
    normalVect_x_directions = np.zeros(len(edges[MARKER_Inside]))
    normalVect_y_directions = np.zeros(len(edges[MARKER_Inside]))

    for i in range(len(edges[MARKER_Inside])) :
        x1, y1 = coord[int(edges[MARKER_Inside][i][0]-1)]
        x2, y2 = coord[int(edges[MARKER_Inside][i][1]-1)]
        vect = [(x2-x1), (y2-y1), 0]
        if(np.sqrt(x2*x2 + y2*y2) > np.sqrt(x1*x1 + y1*y1)) :
            normalVect = np.cross(vect, [0, 0, 1])
        else :
            normalVect = np.cross(vect, [0, 0, -1])

        # Direction of normal vector
        normalVect_x_directions[i], normalVect_y_directions[i] = [normalVect[0]/np.sqrt(normalVect[0]**2+normalVect[1]**2), normalVect[1]/np.sqrt(normalVect[0]**2+normalVect[1]**2)]

        # Coordinates of normal vector
        normalVect_x_coords[i], normalVect_y_coords[i] = [(x1+x2)/2.0, (y1+y2)/2.0]

    if plot :
        # Plots the normal vectors
        cfv.draw_geometry(
                NozzleGeom(),
                label_curves=False,
            )
        plt.quiver(normalVect_x_coords, normalVect_y_coords, normalVect_x_directions, normalVect_y_directions, color = 'r')
        plt.title("Surface normals for the inside of the rocket nozzle")
        plt.xlabel("x-coordinate [m]")
        plt.ylabel("y-coordinate [m]")
        plt.show()
   
    ptype = 2
    ep = np.array([ptype, thickness])
    n_dofs = np.size(dofs)
    ex, ey = cfc.coordxtr(edof, coord, dofs)

    # Calculates the D-matrices for the mechanical problem
    DCu = calcD_matrix(ECu, VCu)
    DTi = calcD_matrix(ETi, VTi)

    # Creates empty K and F-matrices
    K = np.zeros([n_dofs, n_dofs])
    F = np.zeros([n_dofs, 1])

    for i in range(len(edof)):
        # Computes element stiffness matrix 
        if element_markers[i] == MARKER_TiAlloy:
            Ke = cfc.plante(ex[i], ey[i], ep, DTi)
        else:
            Ke = cfc.plante(ex[i], ey[i], ep, DCu)
        
        dof = edof[i]

        # Finds the average temperature of the element
        tAvg = (temps[int(dof[1] / 2 - 1)] + temps[int(dof[3] / 2 - 1)] + temps[int(dof[5] / 2 - 1)]) / 3
        
        # Calculates difference from starting temperature
        deltaT = tAvg[0, 0] - Tinfty

        # Computes contribution due to thermal expansion
        if element_markers[i] == MARKER_TiAlloy:
            epsilon_deltaT = AlphaTi * deltaT * np.array([[1], [1], [0]])
            es = calcD_matrix(ETi, VTi) @ epsilon_deltaT 
        else:
            epsilon_deltaT =  AlphaCu * deltaT * np.array([[1], [1], [0]])
            es = calcD_matrix(ECu, VCu) @ epsilon_deltaT 

        # Computes element load vector
        Fe = cfc.plantf(ex[i], ey[i], ep, es.T)

        # Assembles the element matrices into the global matrices
        cfc.assem(edof[i], K, Ke, F, Fe)

    # Apply Dirichlet boundary conditions 
    bc, bc_value = np.array([], 'i'), np.array([], 'f')
    bc, bc_value = cfu.applybc(bdofs, bc, bc_value, MARKER_ChamberOutside, 0.0)
    bc, bc_value = cfu.applybc(bdofs, bc, bc_value, MARKER_QN_0, 0.0, 2)
    
    # Solves for the displacement due to thermal 
    u_thermal, _ = cfc.solveq(K, F, bc, bc_value)

    # Plots the displacement 
    cfv.figure()
    cfv.draw_displacements(u_thermal, coord, edof, 2, 2, draw_undisplaced_mesh=True, magnfac=100,
                       title="Displacement due to thermal expansion,\n with a magnification factor of 100")
    plt.xlabel("x-coordinate [m]")
    plt.ylabel("y-coordinate [m]")
    plt.show()
    
    # Force from the pressure on the inside of the rocker nozzle
    for i, edge in enumerate(edges[MARKER_Inside]):
        n1, n2 = edge
        L = np.linalg.norm(coord[int(n1) - 1] - coord[int(n2) - 1])

        x_dir = normalVect_x_directions[i]
        y_dir = normalVect_y_directions[i]
        
        normalVect = np.array([x_dir, y_dir])/np.linalg.norm((x_dir, y_dir))
        
        # Converts p_0 into a force (p_0 * area)
        p_0_force = p_0 * thickness * L * normalVect

        # Insert force into load vector
        F[2 * (int(n1) - 1)] -= p_0_force[0]
        F[2 * (int(n1) - 1) + 1] -= p_0_force[1]

    # Solves the full mechancical problem
    u, r = cfc.solveq(K, F, bc, bc_value)

    # Extract element displacements from u
    ed = cfc.extract_eldisp(edof, u)  

    # Calculate von mises stress in each element
    von_mises_per_el = np.zeros(len(edof))

    for i in range(len(edof)):
        dof = edof[i]
        
        # Finds the average temperature of the element
        temp = (temps[int(dof[1] / 2 - 1)] + temps[int(dof[3] / 2 - 1)] + temps[int(dof[5] / 2 - 1)]) / 3
        
         # Calculates difference from starting temperature
        deltaT = temp[0, 0] - Tinfty

        if element_markers[i] == MARKER_CuCr:
            es, _ = cfc.plants(ex[i], ey[i], ep, DCu, ed[i])

            # Computes contribution due to thermal expansion
            epsilon_deltaT = AlphaCu * deltaT * np.array([[1], [1], [0]])
            D_epsilon_deltaT = calcD_matrix(ECu, VCu) @ epsilon_deltaT

            # Calculate stresses needed for sigma_eff
            sigma_x = es[0, 0] - D_epsilon_deltaT[0]
            sigma_y = es[0, 1] - D_epsilon_deltaT[1]
            tau_xy = es[0, 2] - D_epsilon_deltaT[2]
            sigma_z = VCu*(sigma_x + sigma_y) - AlphaCu * ECu * deltaT
        else:
            es, _ = cfc.plants(ex[i], ey[i], ep, DTi, ed[i])

            # Computes contribution due to thermal expansion
            epsilon_deltaT = AlphaTi * deltaT * np.array([[1], [1], [0]])
            D_epsilon_deltaT = calcD_matrix(ETi, VTi) @ epsilon_deltaT

            # Calculate stresses needed for sigma_eff
            sigma_x = es[0, 0] - D_epsilon_deltaT[0]
            sigma_y = es[0, 1] - D_epsilon_deltaT[1]
            tau_xy = es[0, 2] - D_epsilon_deltaT[2]
            sigma_z = VTi * (sigma_x + sigma_y) - AlphaTi * ETi * deltaT
        
        # Calculates sigma_eff
        sigma_eff = np.sqrt(sigma_x **2 + sigma_y **2 + sigma_z **2 - sigma_x*sigma_y - sigma_x*sigma_z - sigma_y*sigma_z + 3 * tau_xy **2)
        von_mises_per_el[i] = sigma_eff[0]
    
    # Computes the old edof-matrix and converts the von Mises element values to nodal values
    _, oldEdof, _, _, _ = MakeThermMesh(NozzleGeom())
    von_mises_nodal_values = elmToNode(von_mises_per_el, oldEdof)
   
   # Prints the maximum von Mises stress
    print(f"Maximum von Mises stress: {np.max(von_mises_nodal_values) / 1e6} MPa")

    if plot :
        # Plots the nodal von Mises stress in the entire rocket nozzle
        cfv.figure(fig_size=(10, 5))
        cfv.draw_nodal_values_shaded(von_mises_nodal_values, coord, oldEdof, "von Mises stress distribution at t = 1 h")
        cfv.colorbar()
        plt.xlabel("x-coordinate [m]")
        plt.ylabel("y-coordinate [m]")
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

def calcD_matrix(E, v) -> np.matrix:
    """
    Computes D-matrix for the mechanical problem 
    according to the constitutive model for plane strain

    Inputs:
        E: Youngs modulus
        v: Poissons ratio

    Outputs:
        D: Constitutive matrix
    """
    D = E * np.array(
        [[1-v, v, 0],
         [v, 1-v, 0],
         [0, 0, (1 - 2 * v) / 2]]
    ) / ((1 + v) * (1 - 2 * v))
    return D

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


def NozzleGeom() :
    """
    Defines the geometry of the rocket nozzle

    Inputs:
        None

    Outputs:
        g: Nozzle geometry
    """
    # Adds the points of the geometry
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

    # Adds edges and marks them
    g.spline([0, 1], 0, marker=MARKER_ChamberOutside)
    g.spline([1, 12], 1, marker=MARKER_ChamberOutside)
    g.spline([12, 2], 10, marker=MARKER_BellOutside)
    g.ellipse([2, 6, 11, 3], 2, marker=MARKER_BellOutside) 
    g.spline([3, 4], 3, marker=MARKER_BellOutside)
    g.ellipse([5, 6, 11, 4], 4, marker=MARKER_Inside)
    g.spline([5, 7], 5, marker=MARKER_Inside)
    g.circle([7, 11, 8], 6, marker=MARKER_Inside)
    g.spline([8, 9], 7, marker=MARKER_Inside)
    g.spline([9, 10], 8, marker=MARKER_Inside)
    g.spline([10, 0], 9, marker=MARKER_QN_0)
    g.spline([12, 8], 11, marker=MARKER_Material_Transition)

    # Adds the two surfaces and marks them 
    g.surface([0, 1, 11, 7, 8, 9], ID=0, marker=MARKER_TiAlloy)
    g.surface([10, 2, 3, 4, 5, 6, 11], ID=1, marker=MARKER_CuCr)
    return g

def MakeThermMesh(geom) :
    """
    Generates a mesh for the thermal problem

    Inputs:
        geom: Geometry for the problem

    Returns relevant lists for the mesh
    """
    mesh = cfm.GmshMeshGenerator(geom, mesh_dir=mesh_dir)

    dofs_pn = 1 # temp
    mesh.el_size_factor = el_sizef
    mesh.el_type = el_type
    mesh.dofs_per_node = dofs_pn

    coord, edof, dofs, bdofs, element_markers = mesh.create()

    return (coord, edof, dofs, bdofs, element_markers)

def AssembleThermStiffness(coord, edof, dofs, element_markers) :
    """
    Calculates and assembles the K-matrix for the thermal problem

    Inputs:
        coord, edof, dofs, element_markers: Mesh properties

    Outputs:
        K: Global stiffness matrix
    """
    ep = [thickness] 

    n_dofs = np.size(dofs) # Number of nodes
    ex, ey = cfc.coordxtr(edof, coord, dofs) # Nodal coordinates

    DCu = np.array([[KCu, 0.0], [0.0, KCu]])  
    DTi = np.array([[KTi, 0.0], [0.0, KTi]])  

    # Initializes empty global stiffness matrix
    K = np.zeros([n_dofs, n_dofs])

    for i in range(len(edof)):
        # Computes element stiffness matrix Ke
        if element_markers[i] == MARKER_TiAlloy:
            Ke = cfc.flw2te(ex[i], ey[i], ep, DTi)
        else:
            Ke = cfc.flw2te(ex[i], ey[i], ep, DCu)
        
        # Assembles Ke it into global stiffness matrix K
        K = cfc.assem(edof[i], K, Ke)

    return K

def MakeThermBC(coord, edof, dofs, bdofs) :
    """
    Calcualtes boundary conditions for the thermal problem 
    and calculates the modification to the global stiffness matrix due to convectiom

    Inputs:
        coord, edof, dofs, bdofs: Mesh properties

    Outputs:
        F: Load vector
        bc: List of edges affected by boundary contition
        bc_value: Values for the boundary conitions
        KModifier: Modification to global stiffness matrix due to convection  
    """
    # Initializes the load vector
    F = np.zeros([np.size(dofs), 1])

    # Finds edges given nodes 
    edges = nodesToEdges(bdofs, edof)

    bc, bc_value = np.array([], 'i'), np.array([], 'f')

    # Add heating boundary condition to load vector
    if MARKER_Inside in bdofs.keys():
        for e in edges[MARKER_Inside] :
            x1, y1 = coord[e[0]-1]
            x2, y2 = coord[e[1]-1]

            dx = x1-x2
            dy = y1-y2
            l = np.sqrt(dx*dx + dy*dy)
            
            # Insert values into load vector
            F[e[0]-1] += l*thickness*ChamberHeating/2
            F[e[1]-1] += l*thickness*ChamberHeating/2

    # Add cooling boundary condition to load vector
    if MARKER_ChamberOutside in bdofs.keys():
        for e in edges[MARKER_ChamberOutside] :
            x1, y1 = coord[e[0]-1]
            x2, y2 = coord[e[1]-1]

            dx = x1-x2
            dy = y1-y2
            l = np.sqrt(dx*dx + dy*dy)

            # Insert values into load vector
            F[e[0]-1] -= l*thickness*ChamberCooling/2
            F[e[1]-1] -= l*thickness*ChamberCooling/2

    # Initializes the modification to the global stiffness matrix
    KModifier = np.zeros([len(F), len(F)])

    # Add convecton boundary condition to load vector and to modification matrix
    if MARKER_BellOutside in bdofs.keys():
        for e in edges[MARKER_BellOutside] :
            x1, y1 = coord[e[0]-1]
            x2, y2 = coord[e[1]-1]

            dx = x1-x2
            dy = y1-y2
            l = np.sqrt(dx*dx + dy*dy)
            
            # Insert values into load vector
            F[e[0]-1] += l*thickness*AlphaConvection*Tinfty/2 
            F[e[1]-1] += l*thickness*AlphaConvection*Tinfty/2
            
            # Insert values into modification matrix
            KModifier[e[0]-1][e[0]-1] += l * thickness * AlphaConvection /3
            KModifier[e[1]-1][e[1]-1] += l * thickness * AlphaConvection /3
            KModifier[e[0]-1][e[1]-1] += l * thickness * AlphaConvection /6
            KModifier[e[1]-1][e[0]-1] += l * thickness * AlphaConvection /6

    return F, bc, bc_value, KModifier

def MakeCapacityMatrix(coord, dofs, edof, element_markers) -> np.array :
    """ 
    Calculates the capacity matrix used in the numerical integration method
    
    Input:
    coord, dofs, edof, element_markers: Mesh properties

    Output: 
        C: Global capacity matrix
    """
    
    # Creates an empty capacity matrix
    C = np.zeros((len(dofs), len(dofs)))

    # Iterates over all elements to calculate the global capacity matrix
    for i in range(len(edof)) :
        xCord = np.zeros(3)
        yCord = np.zeros(3)

        # Finds the coordinates of the nodes in an element
        for j in range(len(edof[i])):
            x, y = coord[edof[i][j]-1]
            xCord[j] += x
            yCord[j] += y

        # Computes the element capacity matrix
        if element_markers[i] == MARKER_CuCr :
            const = thickness*CCu*RhoCu
            Ce = plantml(xCord, yCord, const)
        else:
            const = thickness*CTi*RhoTi
            Ce = plantml(xCord, yCord, const)
        
        # Inserts the element capacity matrix into the global capacity matrix
        for k in range(0,3) : 
            for l in range(0,3) :
                C[edof[i][k]-1][edof[i][l]-1] += Ce[k][l]

    return C

if __name__=="__main__":
    statTherm(plot=True)

    temps = dynTherm(plot=True)

    Mech(plot=True, temps=temps)