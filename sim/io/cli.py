import argparse
from datetime import datetime
from sim.models.linear_cg import linear_cg
from sim.models.linear_dg import linear_dg
from sim.models.fp_coupled import fp_coupled
from sim.models.fp_decoupled import fp_decoupled
from sim.models.nonlin_cg import nonlin_cg

# EXPERIMENTS
from sim.experiments.annihilation_2 import annihilation_2
from sim.experiments.smooth import smooth
from sim.experiments.unstable import unstable


EXPERIMENT_REGISTRY = {
    "smooth": smooth,
    "annihilation" : annihilation_2,
    "unstable" : unstable,
}

MODEL_REGISTRY = {
    "linear_cg": linear_cg,
    "linear_dg": linear_dg,
    "fp_coupled" : fp_coupled,
    "fp_decoupled" : fp_decoupled,
    "nonlin_cg" : nonlin_cg,
}

MODEL_PARAMS = {
    "projection_step": {"type": bool, "default": True, "help": "Apply nodal projection onto unit-sphere after every iteration. Only applicable to: linear_cg, linear_dg."},
    "mass_lumping": {"type": bool, "default": True, "help": "Apply mass-lumping, i.e. vertex based quadrature. Only applicable to: linear_cg."},
    "project_tangent_map": {"type": bool, "default": True, "help": "Apply nodal projection onto the director of the tangent mapping: I - d x d. Only applicable to: linear_cg, linear_dg."},
    "alpha" : {"type": float, "default": 0.1, "help": "Jump stabilization parameter of the discontinuous discrete Laplacian. Only applicable to: linear_dg."},
    "fp_a_tol": {"type": float, "default": 1E-11, "help" : "Absolute tolerance for a fixed point solver."},
    "fp_r_tol": {"type": float, "default": 1E-10, "help" : "Relativ tolerance for a fixed point solver."},
    "fp_max_iters": {"type": int, "default": 100, "help" : "Maximum inner iterations for a fixed point solver."},
    "seed": {"type": int, "default": 1, "help" : "Random seed for reproducibility."},
}

EXP_SPECIFIC_PARAMS = {
    "dim": {"type": int, "help": "Dimension of the experiment, e.g. 2 for 2D or 3 for 3D.",
            "defaults": { 
                "smooth" : 2 ,
                "annihilation" : 3,
                "stb3d" : 3,
                "unstable" : 2,
                }
            },
    "dt": {"type": float, "help": "Time step size for the simulation.",
           "defaults": 
           { 
               "smooth": 0.01 ,
               "annihilation" : 0.01,
                "stb3d" : 0.01,
                "unstable" : 0.01,
               }
            },
    "dh": {"type": int, "help": "Spatial resolution, i.e. number of elements in the mesh.",
           "defaults": {
               "smooth": 40,
               "annihilation" : 5,
                "stb3d" : 5,
                "unstable" : 10,
               }
            },
    "T": {"type": float, "help": "Total simulation time.",
          "defaults": {
                "smooth": 1.0,
                "annihilation" : 0.5,
                "stb3d" : 1.0,
                "unstable" : 1.0,
            }
        },
}

SOLVER_PARAMS = {
    "ksp_type_d": {
                "type" : str, 
                "default" : "preonly", 
                "help" : "Krylov Subspace Method (KSP) used for the director equation."
            },
    "pc_type_d" : {
                "type" : str, 
                "default" : "jacobi", 
                "help" : "Preconditioner (PC) used for the director equation."
            },
        
    "ksp_type_q": { 
                "type" : str, 
                "default" : "preonly", 
                "help" : "Krylov Subspace Method (KSP) used for the energy variational equation."
            },
    "pc_type_q" : {
                "type" : str, 
                "default" : "sor", 
                "help" : "Preconditioner (PC) used for the energy variational equation."
            },
    "pc_factor_mat_solver_type" : {
                "type" : str, 
                "default" : "mumps", 
                "help" : "Solver package used to perform the factorization." 
            },
    "atol" : {
                "type" : float, 
                "default" : 1.0e-11, 
                "help" : "Absolute Tolerance of nonlinear solver."
            },
    "rtol" : {
                "type" : float, 
                "default" : 1.0e-10, 
                "help" : "Absolute Tolerance of nonlinear solver."
            },
    "ksp_atol" : {
                "type" : float, 
                "default" : 1.0e-12, 
                "help" : "Absolute Tolerance of nonlinear solver."
            },
    "ksp_rtol" : {
                "type" : float, 
                "default" : 1.0e-11, 
                "help" : "Absolute Tolerance of nonlinear solver."
            },
    "n_max_it": {
                "type" : int, 
                "default" : 100, 
                "help" : "Maximum number of iterations for Newton solver."
            },
    "ksp_type_n": {
                "type" : str, 
                "default" : "gmres", 
                "help" : "Krylov Subspace Method (KSP) used for the Newton Solver. Default: fgmres. Recommended options: cg, gmres, fgmres."
            },
    "pc_type_n" : {
                "type" : str, 
                "default" : "gamg", 
                "help" : "Preconditioner (PC) used for the Newton Solver. Default: gamg. Recommended options: gamg, hypre."
            },
}

HEAT_FLOW_PARAMS = {
    "gamma": {"type": float, "default": 1.0, "help": "Physical parameter, see README."},
    "K1": {"type": float, "default": 1.0, "help": "Physical parameter in the Oseen-Frank energy, see README. REQUIRED, MUST BE POSITIVE."},
    "K2": {"type": float, "default": 0.0, "help": "Physical parameter in the Oseen-Frank energy, see README."},
    "K3": {"type": float, "default": 0.0, "help": "Physical parameter in the Oseen-Frank energy, see README. Only well-defined in three dimensions."},
    "K4": {"type": float, "default": 0.0, "help": "Physical parameter in the Oseen-Frank energy, see README. Only well-defined in three dimensions."},
    "K5": {"type": float, "default": 0.0, "help": "Physical parameter in the Oseen-Frank energy, see README. Only well-defined in three dimensions."},
    "chi_vert": {"type": float, "default": 0.0, "help": "Physical parameter for the coupling with the magnetic field, see README. MUST BE NEGATIVE."},
    "chi_perp": {"type": float, "default": 0.0, "help": "Physical parameter for the coupling with the magnetic field, see README. MUST BE NEGATIVE."},
    "lam": {"type": int, "default": 1.0, "help": "Choice of parameter for Hilbert-Schmidt Operator Q, see README. Choice of Lambda: 1 = constant set to 1.0, 2 = 2^-l, 3 = (2*l^2)^(-l)."},
    "cs": {"type": float, "default": 1.0, "help": "Coupling/Scaling parameter for the stochastic noise term, see README."},
}

OUTPUT_PARAMS = {
    "folderpath": {"flags": ["-fp", "--folderpath"], "type": str, "default": "", "help": "Define the path to the folder for the output of the simulations."},
    "sim_id": {"flags": ["-sid", "--sim_id"], "type": str, "default": "", "help": "Overwrite the name of the simulation folder. Otherwise a unique identifier is chosen."},
    "xdmf": {"type": bool, "default": False, "help": "Store FEM functions in xdmf format. XDMF Files only work with Lagrange spaces of first order."},
    "vtx": {"type": bool, "default": False, "help": "Store FEM functions in vtx (.bp) format. VTX Files are very flexible with respect to the FEM space."},
    "vtk": {"type": bool, "default": False, "help": "Store  FEM functions in vtk (.pvd) format. VTK Files only work with Lagrange spaces of order <= 2 or DG1 spaces."},
    "checkpoint": {"flags": ["-cp", "--checkpoint"], "type": bool, "default": False, "help": "Checkpoint FEM functions to compare later."},
    "overwrite": {"flags": ["-ovw", "--overwrite"], "type": bool, "default": False, "help": "Overwrites existing data, if a simulation with the given ID already exists.."},
    "FunctionSaveRate": {"flags": ["-fsr", "--functionsaverate"], "type": float, "default": 0.1, "help": "Save frequency of functions. The input 0.1 saves the function every 10 percent of the temporal evolution --> 11 saving points in time."},
    "MetricSaveRate": {"flags": ["-msr", "--metricsaverate"], "type": float, "default": 0.05, "help": "Save frequency of function metrics such as Energy. The input 0.05 saves the function every 5 percent of the temporal evolution --> 21 saving points in time."},
}

TGUI_PARAMS = {
    "theight": { "type": int, "default": (3+18+7+15), "help": "Manually prescribe height in terminal GUI."},
    "twidth": { "type": int, "default": 200, "help": "Manually prescribe width in terminal GUI."},
}


def add_param_group(parser, param_dict, description):
    group = parser.add_argument_group(description)
    for name, param in param_dict.items():
        if param.get("type","") == bool:
            param["type"] = int
            param["default"] = int(param["default"])
            param["choices"] = [0,1]
        flags = param.get("flags", [f"-{name}"])
        if param.get("default",None) != None:
            help_str = param.get("help", "")+" Default value: "+str(param.get("default", ""))
        else:
            help_str = param.get("help", "")
        group.add_argument(*flags, type=param["type"], default=param.get("default", None), choices= param.get("choices",None), help=help_str)

def parse_args():
    parser = argparse.ArgumentParser(description="ericksen_leslie_x\nFEniCSx implementation of numerical methods for the Ericksen--Leslie equations for nematic liquid crystal flow. ")
    
    parser.add_argument('-m','--mod', type=str, required=True, choices=MODEL_REGISTRY.keys(), help='THIS ARGUMENT IS REQUIRED: The model considered in the simulation.' ) 
    parser.add_argument('-e','--exp', type=str, required=True, choices=EXPERIMENT_REGISTRY.keys(), help='THIS ARGUMENT IS REQUIRED: The experiment setting considered in the simulation.' ) 
    
    add_param_group(parser, MODEL_PARAMS, description = "Parameters for the numerical model")
    add_param_group(parser, EXP_SPECIFIC_PARAMS, description = "Parameters for the experiment setting")
    add_param_group(parser, HEAT_FLOW_PARAMS, description = "(Physical) Parameters for the Ericksen-Leslie model")        
    add_param_group(parser, SOLVER_PARAMS, description = "Parameters for the PETSC solver")
    add_param_group(parser, OUTPUT_PARAMS, description = "Input and Output settings")    
    add_param_group(parser, TGUI_PARAMS, description = "Terminal GUI settings")

    # Parse
    args = parser.parse_args()
    
    # evtl. overwrite defaults with experiment specific defaults
    for key in EXP_SPECIFIC_PARAMS.keys():
        if hasattr(args, key) and getattr(args, key) is None:
            setattr(args, key, EXP_SPECIFIC_PARAMS[key]["defaults"][args.exp])

    # Set Timestamp
    timestamp = datetime.now().strftime("Y%Y-M%m-D%d_H%H-M%M-S%S")
    setattr(args, "timestamp", timestamp)
    # set simulation ID
    if getattr(args, "sim_id") == "":        
        sim_id= "sim_"+ args.exp +"_"+ args.mod +"_dh"+str(args.dh) +"_dt"+str(args.dt)+"_"+timestamp
        setattr(args, "sim_id", sim_id)

    # Check if the at least one saving format is activated
    if args.xdmf == False and args.vtk == False and args.vtx == False and args.checkpoint == False:
        print("You have selected no format for saving the function. No function data will be saved during the simulation. Check python -m sim -h for options. Do you want to continue? (Press ENTER to continue.)")
        if args.overwrite == 0:
            answer = input()
    
    return args, MODEL_REGISTRY[args.mod], EXPERIMENT_REGISTRY[args.exp]

if __name__ == "__main__":
    args, mod, exp = parse_args()
    print("The following arguments were parsed:\n", args)
    print("Model:", mod)
    print("Experiment:", exp)