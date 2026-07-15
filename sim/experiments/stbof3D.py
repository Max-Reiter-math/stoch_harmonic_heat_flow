from argparse import Namespace
from functools import partial
import numpy as np
from dolfinx.mesh import create_rectangle, create_box, CellType, locate_entities_boundary
from dolfinx.fem import Constant
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from sim.common.meta_bcs import *
from sim.common.mesh import circumcenters, midpoints

"""
Usage examples:
python -m sim.run -m fp_coupled -e stb3d -K1 0.0001 -K3 1.0 -vtx 1 -sid splay -dt 0.01 -fsr 0.1
python -m sim.run -m fp_coupled -e stb3d -K1 0.0001 -K2 1.0 -K5 1.0 -vtx 1 -sid twist -dt 0.01 -fsr 0.1
python -m sim.run -m fp_coupled -e stb3d -K1 0.0001 -K2 1.0 -K4 1.0 -vtx 1 -sid bend -dt 0.01 -fsr 0.1
"""

class stb3D:
    def __init__(self, comm, args = Namespace()):
        # NAME
        self.name="Splay Twist Bend bcs 3D"
        self.dim = args.dim

        # MESH
        if args.mod in ["linear_dg"]:
            celltype = CellType.hexahedron
        else:
            celltype = CellType.tetrahedron

        self.mesh = create_box(comm, [np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0,1.0])],  [args.dh,args.dh,args.dh], cell_type = celltype)
        self.boundary = boundary_3d

        # MESHTAGS
        # entities locate_entities_boundary(self.mesh, self.dim-1, self.boundary)
        self.meshtags = None

        #DG0 int points
        if args.mod in ["linear_dg"]:
                self.dg0_cells, self.dg0_int_points = midpoints(self.mesh)

        # INIT FUNCTIONS WRT DIMENSION
        no_slip = partial(get_no_slip, dim = 3)

        # INITIAL CONDITIONS
        self.initial_conditions = {"v": no_slip, "p": (lambda x: np.full((x.shape[1],), 0.0)), "d": d0}

        # BOUNDARY CONDITIONS
        self.boundary_conditions = [meta_dirichletbc("v", "geometrical", no_slip,  marker = self.boundary)]
        self.boundary_conditions += [meta_dirichletbc("d", "geometrical", partial(twist, axis = 1), marker = yborders)] 
        self.boundary_conditions += [meta_dirichletbc("d", "geometrical", partial(bend, axis = 2), marker = zborders)] 
        self.boundary_conditions += [meta_dirichletbc("d", "geometrical", partial(splay, axis = 0), marker = xborders)] 
    
    @property
    def info(self):
        return {"name":self.name}
    @property
    def has_exact_solution(self):
        return False

def boundary_3d(x: np.ndarray) -> np.ndarray:
    return np.logical_or.reduce((np.isclose(x[0], 0.0), np.isclose(x[0], 1.0), np.isclose(x[1], 0.0), np.isclose(x[1], 1.0),np.isclose(x[2], 0.0), np.isclose(x[2], 1.0)))

def xborders(x: np.ndarray) -> np.ndarray:
    return np.logical_or.reduce((np.isclose(x[0], 0.0), np.isclose(x[0], 1.0)))

def yborders(x: np.ndarray) -> np.ndarray:
    return np.logical_or.reduce((np.isclose(x[1], 0.0), np.isclose(x[1], 1.0)))

def zborders(x: np.ndarray) -> np.ndarray:
    return np.logical_or.reduce((np.isclose(x[2], 0.0), np.isclose(x[2], 1.0)))

def splay(x: np.ndarray, axis: int)-> np.ndarray:
    # x hase shape (dimension, points)
    values = np.zeros((3, x.shape[1])) # values is going to be the output

    viewpoint_axis1 = -0.5
    viewpoint_axis2 = 0.5

    axis1 = axis
    axis2 = (axis+1)%3

    values[axis1] = x[axis1] - viewpoint_axis1
    values[axis2] = x[axis2] - viewpoint_axis2

    # renormalization
    norms = np.linalg.norm(values, ord = 2, axis = 0) # compute euclidean norm
    values = values / norms # renormalize
    return values

def twist(x: np.ndarray, axis: int)-> np.ndarray:
    # x hase shape (dimension, points)
    values = np.zeros((3, x.shape[1])) # values is going to be the output

    mode = axis #0,1,2

    values[ (mode-1) % 3]= 1.0*(x[mode % 3]) 
    values[(mode + 1) % 3]= 1.0*(1.0-x[mode % 3])

    # renormalization
    norms = np.linalg.norm(values, ord = 2, axis = 0) # compute euclidean norm
    values = values / norms # renormalize
    return values

def bend(x: np.ndarray, axis: int)-> np.ndarray:
    # x hase shape (dimension, points)
    values = np.zeros((3, x.shape[1])) # values is going to be the output

    mode = axis #0,1,2

    values[mode % 3]= np.cos(np.pi*x[(mode + 1) % 3])
    values[(mode + 1) % 3]= np.sin(np.pi*x[(mode + 1) % 3])

    return values

def d0(x: np.ndarray)-> np.ndarray:
    # x hase shape (dimension, points)
    values = np.zeros((3, x.shape[1])) # values is going to be the output

    # K_1         = 0
    # K_2         = 1
    # K_3         = 1
    # values = K_1*splay(x,axis=0)+K_2*twist(x,axis=1)+K_3*bend(x,axis=2)

    np.random.seed(20250909)

    theta = np.pi + np.random.rand(x.shape[1])*np.pi # this way all directors are only in a 180 degree radius
    phi = np.pi + np.random.rand(x.shape[1])*2*np.pi # this way all directors are only in a 360 degree radius

    values[0] = np.sin(theta)*np.cos(phi)
    values[1] = np.sin(theta)*np.sin(phi)
    values[2] = np.cos(theta)
    
    # x hase shape (dimension, points)
    # values = np.zeros((3, x.shape[1])) # values is going to be the output
    # values[0]= 1.0
    # values[1]= 1.0
    # values[2]= 1.0

    # renormalization
    # norms = np.linalg.norm(values, ord = 2, axis = 0) # compute euclidean norm
    # values = values / norms # renormalize
    return values
    
def get_no_slip(x: np.ndarray, dim: int) -> np.ndarray:
    # x hase shape (dimension, points)
    if dim >1:
        values = np.zeros((dim, x.shape[1]))
    else: values = np.zeros(x.shape[1])
    return values
