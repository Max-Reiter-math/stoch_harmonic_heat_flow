
from pathlib import Path
from mpi4py import MPI
from ufl import dx, inner, grad
from dolfinx.fem import functionspace, Function, ElementMetaData, create_interpolation_data
import numpy as np
import adios4dolfinx
from sim.common.norms import L2_norm, H10_norm, H1_norm

def compare(comm, filename1: Path | str, quantity1: str, family1: str, degree1: int, filename2: Path | str, quantity2: str, family2: str, degree2: int, family_comp: str, degree_comp: int, shape = (1,), space_norm = "L2", time_norm = "L2"):
    """
    Compares the checkpoints of two functions based on the shared time partition.
    The FIRST file should always include the finer mesh. The second one should be saved on a submesh.
    """
    time_partition1 = adios4dolfinx.read_timestamps(filename1, comm, quantity1, engine='BP4')
    time_partition2 = adios4dolfinx.read_timestamps(filename2, comm, quantity2, engine='BP4')
    # interesect time partitions
    # TODO - rounding necessary?
    time_partition  = np.intersect1d(time_partition1,time_partition2)
    
    u1              = create_function(comm, filename1, family1, degree1, shape = shape, name = quantity1)
    u2              = create_function(comm, filename2, family2, degree2, shape = shape, name = quantity2)
    domain = u1.function_space.mesh
    FS_comp = functionspace(domain, ElementMetaData(family_comp, degree_comp, shape=shape))

    #NOTE - Since the mesh sequence is hierarchical we have to interpolate the approximate solutions on the coarser meshes onto the FEM space on the finer meshes
    # int_data = create_nonmatching_meshes_interpolation_data(u2_higher_res.function_space.mesh._cpp_object, u2_higher_res.function_space.element, u2.function_space.mesh._cpp_object, 0.0) # padding seems to be some tolerance variable for the collision detection. So we set it to 0.0 on standard
    u1_comp     = Function(FS_comp)
    u2_comp     = Function(FS_comp)
    # Check that both interfaces of create nonmatching meshes interpolation data returns the same
    
    fine_mesh_cell_map = domain.topology.index_map(domain.topology.dim)
    num_cells_on_proc = fine_mesh_cell_map.size_local + fine_mesh_cell_map.num_ghosts
    cells = np.arange(num_cells_on_proc, dtype=np.int32)

    int_data1 = create_interpolation_data(u1_comp.function_space, u2.function_space, cells)
    int_data2 = create_interpolation_data(u2_comp.function_space, u2.function_space, cells)
    
    errors_by_time = []

    for time in time_partition:
        adios4dolfinx.read_function(filename1, u1, time=time, name = quantity1)
        adios4dolfinx.read_function(filename2, u2, time=time, name = quantity2)        
        
        u1_comp.interpolate_nonmatching(u1, cells, int_data1)
        u2_comp.interpolate_nonmatching(u2, cells, int_data2)

        # Integrate the error
        e = u1_comp - u2_comp

        if space_norm == "L2":
            err = L2_norm(comm, e)
        elif space_norm == "H10":
            err = H10_norm(comm, e)
        elif space_norm == "H1":
            err = H1_norm(comm, e)

        errors_by_time.append(err)
    print(errors_by_time)
    if time_norm == "L2":
        final_err = np.sum(np.array(errors_by_time)**2)**0.5
    elif time_norm == "inf":
        final_err = np.max(errors_by_time)
    return final_err









def create_function(comm, filename: Path, family: str, degree: int, shape = (1,), name = None):
    domain  = adios4dolfinx.read_mesh(filename, comm, engine = "BP4")
    el      = ElementMetaData(family, degree , shape=shape)
    FS      = functionspace(domain, el)
    f       = Function(FS)
    return f

def get_timestamps(path):
    import adios2
    timestamps = []
    with adios2.open(path, "r", MPI.COMM_SELF) as fh:
        for fstep in fh:
            # inspect variables in current step
            step_vars = fstep.available_variables()
            t = fstep.read("f_time")
            timestamps.append(t)

        return timestamps
    
if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    # TODO - process large number of structured function comparison
    quantity = "d"
    family_comp, degree_comp = "P", 1
    
    filename_ref = "output/cp1/cp-d.bp"

    filename = "output/cp2/cp-d.bp"
    err = compare(comm, filename_ref, "d", "Lagrange", 1, filename, "d", "Lagrange", 1, family_comp, degree_comp, shape = (2,), space_norm = "L2")
    err = np.round(err, decimals=4)
    print(err)
    print("--")
    filename = "output/cp3/cp-d.bp"
    err = compare(comm, filename_ref, "d", "Lagrange", 1, filename, "d", "Lagrange", 1,  family_comp, degree_comp, shape = (2,), space_norm = "L2")
    err = np.round(err, decimals=4)
    print(err)
    print("--")
    filename = "output/cp4/cp-d.bp"
    err = compare(comm, filename_ref, "d", "Lagrange", 1, filename, "d", "Lagrange", 1,  family_comp, degree_comp, shape = (2,), space_norm = "L2")
    err = np.round(err, decimals=4)
    print(err)
    print("--")
    
    

    # filename2 = "output/cp2/cp-d.bp"
    # err = compare(comm, filename_ref, "d", "Lagrange", 1, filename2, "d", "Lagrange", 1, "DG", 1, shape = (2,), space_norm = "H1")
    # err = np.round(err, decimals=4)
    # print(err)
    # print("--")
    # filename3 = "output/cp3/cp-d.bp"
    # err = compare(comm, filename_ref, "d", "Lagrange", 1, filename3, "d", "DG", 1, "DG", 1, shape = (2,), space_norm = "H1")
    # err = np.round(err, decimals=4)
    # print(err)