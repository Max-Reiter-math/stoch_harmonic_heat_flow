from functools import partial
from ufl import grad, div, curl, inner, dot, cross, dx, Measure, FacetNormal, TrialFunction, TestFunction
import dolfinx.la as la
from dolfinx.fem import Function, functionspace, form, assemble_scalar, ElementMetaData, bcs_by_block, extract_function_spaces, bcs_by_block
from dolfinx.fem.petsc import assemble_matrix, assemble_matrix_nest, assemble_vector_nest, apply_lifting_nest, set_bc_nest, set_bc_nest, LinearProblem
from sim.common.operators import *
from sim.common.common_fem_methods import *
from sim.common.meta_bcs import *
from sim.common.stochastics import stratonovich_noise, get_lambda_method

def linear_cg(comm, experiment, args, postprocess=None):
    """
    Algorithm 1 (setting the velocity to zero in time and space) by Maximilian E. V. Reiter. (2025). Projection Methods in the Context of Nematic Crystal Flow.
    
    Important properties of the algorithm:
    - fulfills a discrete energy law
    - unconditional existence
    - automatically fulfills unit-norm constraint
    - assumes a weakly-acute mesh
    """

    if postprocess and comm.rank == 0:
        postprocess.log("dict", "static", {"# MPI Ranks": comm.size})

    # SECTION PARAMETERS
    dim = args.dim
    dt  = args.dt
    t   = 0
    
    if postprocess and comm.rank == 0:
        postprocess.log("dict", "static",{"SID": args.sim_id, "MODEL":args.mod, "PROJECTION STEP": args.projection_step, "PROJECT TANGENT MATRIX": args.project_tangent_map, "MASS LUMPING" : args.mass_lumping, "EXPERIMENT":args.exp, "mesh res.": args.dh, "dt":args.dt, "dim":args.dim, "T": args.T, "seed":args.seed, "cs":args.cs, "approx_method": args.lam})

    #!SECTION PARAMETERS

    domain        = experiment.mesh    
    meshtags      = experiment.meshtags

    #add all local variables for transparency    
    if postprocess and comm.rank == 0:
        postprocess.log("dict","static",{"model.vars":dict(locals())}, visible =False)

    initial_conditions          = experiment.initial_conditions
    boundary_conditions         = experiment.boundary_conditions

    total_time_start = mpi_time(comm)
    
    # SECTION - FUNCTION SPACES AND FUNCTIONS

    D, Y        = functionspace(domain, ElementMetaData("Lagrange", 1 , shape=(dim,))), functionspace(domain, ElementMetaData("Lagrange", 1 , shape=(dim,)))
    TensorF     = functionspace(domain, ElementMetaData("Lagrange", 1 , shape=(dim, dim)))
    
    d1, q1              = TrialFunction(D), TrialFunction(Y)
    d0, d_, q0          = Function(D), Function(D), Function(Y) # d_ is used for the tangent projection
    d, q                = Function(D), Function(Y)
    c_test, b_test      = TestFunction(D), TestFunction(Y)

    # STOCHASTIC NOISE TERM
    noise      = Function(D)    
    get_lambda = get_lambda_method(args.lam)
    noise_eval = partial(stratonovich_noise,dh=args.dh, dt = dt, get_lambda=get_lambda, dim = dim)
    
    # FOR COMPUTATION OF INITIAL q
    qc1 = TrialFunction(Y)
    q_ic = Function(Y)
    bc_test = TestFunction(Y)  

    # COMPUTE AND SAVE DOFS PER RANK
   
    local_dofs = D.dofmap.index_map.size_local + Y.dofmap.index_map.size_local    # Count DOFs on this rank    
    local_info = {f"rank {comm.rank} dofs": local_dofs}                                                                                           # Create local dictionary   
    all_info = comm.gather(local_info, root=0)                                                                                                      # Gather all dictionaries at root

    if comm.rank == 0:
        # Merge all into a single dict
        combined = {}
        for partial_dict in all_info:
            combined.update(partial_dict)
        postprocess.log("dict", "static", combined)

    #!SECTION FUNCTION SPACES AND FUNCTIONS
    
    # SECTION VARIATONAL FORMULATION
    
    if args.mass_lumping:
        dxL = Measure("dx", domain = domain, metadata = {"quadrature_rule": "vertex", "quadrature_degree": 0})
    else:
        dxL = dx

    # SECTION INITIAL CONDITIONS
    d0.interpolate(initial_conditions["d"]) 
    d_.interpolate(initial_conditions["d"]) 
    if "H" in initial_conditions.keys():
        H = Function(D)
        H.interpolate(initial_conditions["H"])
        # TODO - scatter forward
    else:
        H = None
    scatter_all([d0, d_])
    
    # COMPUTATION OF INITIAL DIVERGENCE (BECAUSE WE USE IT TO SET THE BOUNDARY CONDITIONS FOR q)
    problem0 = LinearProblem(form(inner(qc1, bc_test)*dxL), form(q_elastic_energy(args, d0, d0, bc_test, H = None)),  bcs=[], u=q_ic, petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"})
    problem0.solve()
    q0.interpolate(q_ic)
    q0.x.scatter_forward()
    
    #!SECTION

    #SECTION - BOUNDARY CONDITIONS
    bcs = []
    
    for bcwofs in boundary_conditions:
        if bcwofs.type == "Dirichlet":
            if bcwofs.quantity == "d":
                bcwofs.set_fs(D)
                bcs.append(bcwofs.bc)
                # BOUNDARY CONDITIONS FOR AUXILIARY VARIABLE
                #NOTE - We assume that the initial condition for the director field fulfills the boundary conditions imposed
                bcq = meta_dirichletbc("q", bcwofs.find_dofs, q_ic, marker = bcwofs.marker , entity_dim = bcwofs.dim, entities = bcwofs.entities)
                bcq.set_fs( Y)
                bcs.append(bcq.bc)
        # elif bcwofs.type == "Neumann":            
        # elif bcwofs.type == "Robin":
        else: postprocess.log("dict", "static",{"Warning" : "Boundary conditions of type "+bcwofs.type+" are currently not implemented and will be ignored..."} )
    
    #!SECTION

    #SECTION - VARIATIONAL FORM
    a, L = variational_form(dxL, d1, q1, d0, q0, b_test, c_test, d_,  dt, args, H=H, noise = noise)
    #!SECTION
    
    #SECTION - POSTPROCESSING FOR t=0

    if postprocess:
        postprocess.log_functions(0.0, {"d":d0, "q":q0, "n": noise}, mesh = domain) #, meshtags = meshtags)

    metrics = compute_metrics(comm, args, d0,d0,q0, dxL , H = H) # for initial condition
    if postprocess and comm.rank == 0:
        postprocess.log("dict", t, { "time" : t} |  metrics )

    #!SECTION

    # SECTION TIME EVOLUTION
    total_time = mpi_time(comm, start = total_time_start )

    while t < args.T:
        t += dt 
        noise.interpolate(noise_eval)

        # SECTION - ASSEMBLY
        measure_assembly_start = mpi_time(comm)

        # Assemble nested matrix operators
        # NOTE - the following is based on the dolfinx tutorials        
        A = assemble_matrix_nest(a, bcs=bcs)
        A.assemble()

        # Assemble right-hand side vector
        b = assemble_vector_nest(L)

        # Modify ('lift') the RHS for Dirichlet boundary conditions
        apply_lifting_nest(b, a, bcs=bcs)

        # Ghost Updates
        for b_sub in b.getNestSubVecs():
            b_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        # Set Dirichlet boundary condition values in the RHS vector
        bcs0 = bcs_by_block(extract_function_spaces(L), bcs)
        set_bc_nest(b, bcs0)

        assembly_time = mpi_time(comm, start= measure_assembly_start)

        #!SECTION 
        
        #SECTION - SOLVER
        # SETUP SOLVERS
        start_solsetup = mpi_time(comm)
        ksp = setup_split_solver(comm, args, A)
        time_solsetup = mpi_time(comm, start = start_solsetup)

        # The vectors are combined to form a  nested vector and the system is solved.
        x = PETSc.Vec().createNest([la.create_petsc_vector_wrap(d.x), la.create_petsc_vector_wrap(q.x)])

        # SOLVING
        start_sol = mpi_time(comm)
        ksp.solve(b, x)
        time_sol = mpi_time(comm, start = start_sol)

        #!SECTION

        #SECTION - EVTL. METRICS BEFORE PROJECTION STEP  
        if postprocess and comm.rank == 0:
            postprocess.log("dict", t, {"time":t, "t.ass": assembly_time, "t.sol": time_sol, "t.solsetup" : time_solsetup}, visible = True)  
            
        if args.projection_step ==1 or args.project_tangent_map ==1:
            metrics = compute_metrics(comm, args, d,d0,q, dxL, H=H, id=".b4p")
            if postprocess and comm.rank == 0:
                postprocess.log("dict", t, { "time" : t} | metrics , visible = False)

        #!SECTION

        # SECTION - NODAL PROJECTION STEP
        if args.projection_step ==1:
            start_pstep = mpi_time(comm)

            nodal_normalization(d, dim)

            time_pstep = mpi_time(comm, start = start_pstep )
        #!SECTION 

        #SECTION - UPDATE
        update_and_scatter([d0,d_,q0], [d,d,q])
        #!SECTION 

        # SECTION - NODAL PROJECTION STEP FOR TANGENT MAP
        if args.project_tangent_map == 1:
            start_pstep2 = mpi_time(comm)

            nodal_normalization(d_, dim)

            time_pstep2 = mpi_time(comm, start = start_pstep2 )
        #!SECTION 

        
        
        #SECTION - METRICS AT END OF ITERATION 
        metrics =  compute_metrics(comm, args, d,d0,q, dxL , H=H)

        errorL2 = np.nan
        if experiment.has_exact_solution:   
            errorL2 = experiment.compute_error(comm, d,t,norm = "L2", degree_raise = 3)   

        total_time = mpi_time(comm, start = total_time_start )
        if postprocess and comm.rank == 0:
            postprocess.log("dict", t, { "time" : t, "errorL2" : errorL2 , "t.tot" : total_time} | metrics)
            if args.projection_step ==1: postprocess.log("dict", t, { "time" : t, "t.pstep" : time_pstep})
            if args.project_tangent_map == 1: postprocess.log("dict", t, { "time" : t, "t.pstep2" : time_pstep2})
        
        #!SECTION

        #SECTION - SAVING
        if postprocess: 
            postprocess.log_functions(t, {"d":d0, "q":q0, "n": noise}, mesh = domain, meshtags = meshtags)

        #!SECTION

    #!SECTION TIME EVOLUTION

    if postprocess: 
        postprocess.close()

#!SECTION GENERAL METHOD



def variational_form(dxL, d1, q1, d0, q0, b, c, d_, dt, args, H=None, noise = None):
        """
        The variational form is structured as followed:
        a(.,.) : bilinear form describing the lhs of the system
        L(.) : linear form describing the rhs of the system
        a_ij(.,.) : bilinear form describing the lhs of the system depending on the i-th test function and the j-th trial function
        """


        # SECTION DIRECTOR EQUATION
        a_11 = inner(d1, c)*dxL

        if args.gamma != 0.0:
            a_12 = (-1)*dt* args.gamma *inner(q1, a_times_b_times_c(d_,d_,c))*dxL

        L_1 = inner(d0, c)*dxL

        if not( noise is None ):
            if args.dim == 2:
                d_orth = as_vector([-d_[1], d_[0]])
                L_1 += args.cs*inner(noise,d_)*inner(d_orth,c)*dxL
            else:
                L_1 += args.cs*inner(cross(d_, noise),c)*dxL

        #!SECTION DIRECTOR EQUATION

        # SECTION EQUATION FOR THE VARIATIONAL DERIVATIVE
        a_21 = q_elastic_energy(args, d1, d0, b, H = H)

        a_22 = (-1)* inner(q1, b)*dxL 

        zero_b = Function(q0.function_space)
        L_2 = inner(zero_b , b) * dx

        #!SECTION EQUATION FOR THE VARIATIONAL DERIVATIVE
        
        """
        Full bilinear form (lhs) and linear form (rhs)
        """
        a = form([
            [a_11, a_12], 
            [a_21, a_22]
        ])

        L = form([
            L_1 ,
            L_2 ,
        ]) 

        return a, L

def q_elastic_energy(args, d1, d0, b, H = None):
    eq = args.K1 * inner( grad(d1), grad(b))*dx
    if args.K2 != 0.0:
        eq += args.K2 * inner( div(d1), div(b))*dx
    if args.K3 != 0.0:
        eq += args.K3 * inner( curl(d1), curl(b))*dx
    if args.K4 != 0.0 or args.K5 != 0.0:
        raise ValueError("K4 != 0 or K5 != 0 is not supported in the linear scheme.")
        # eq += args.K4 * inner( d1, curl(d1)) * ( inner(d0, curl(b)) + inner(b, curl(d1)) )*dx
        #NOTE - Since the curl is present a simplification to 2D is not possible.
        # eq += args.K5 * inner( cross( d1, curl(d1)) , cross(d0, curl(b)) + cross(b, curl(d1)) )*dx
        #NOTE - The cross product could be replaced by an according tangential matrix. However, since the curl is present a simplification to 2D is not possible anyways.

    if H is not None and args.chi_vert != 0.0:
        eq -= args.chi_vert * inner( d1, H)*inner( b, H)*dx
    if H is not None and args.chi_perp != 0.0:
        eq += args.chi_perp * inner(  H, a_times_b_times_c(d1, b, H) )*dx

    return eq

def setup_split_solver(comm, args, A):
    # Create a nested matrix P to use as the preconditioner. The
    # top-left block of P is shared with the top-left block of A. The
    # bottom-right diagonal entry is assembled from the form a_p11:
    P = PETSc.Mat().createNest([
        [A.getNestSubMatrix(0,0), None],
        [None, A.getNestSubMatrix(1, 1)],
        ])
    P.assemble()

    # Create a MINRES Krylov solver and a block-diagonal preconditioner
    # using PETSc's additive fieldsplit preconditioner
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A, P)
    
    
    # ksp.setOperators(A_ass) #, A_ass)
    ksp.setType("gmres")
    ksp.setTolerances(rtol=1e-9)
    ksp.getPC().setFactorSolverType(PETSc.Mat.SolverType.MUMPS)
    ksp.getPC().setType("fieldsplit")
    ksp.getPC().setFieldSplitType(PETSc.PC.CompositeType.MULTIPLICATIVE)

    # Return the index sets representing the row and column spaces. 2 times Blocks spaces
    nested_IS = P.getNestISs()
    ksp.getPC().setFieldSplitIS(("d", nested_IS[0][0]), ("q", nested_IS[0][1]))

    # Set the preconditioners for each block. This is done via command line interface.
    ksp_d, ksp_q = ksp.getPC().getFieldSplitSubKSP()
    ksp_d.setType(args.ksp_type_d) 
    ksp_d.getPC().setType(args.pc_type_d) 
    ksp_q.setType(args.ksp_type_q) 
    ksp_q.getPC().setType(args.pc_type_q) 

    return ksp

def compute_metrics(comm, args, d,d0,q, dxL, H= None, id =""):
    # ENERGY TERMS
    E_ela1  = assemble_scalar(form(   0.5 *  inner(grad(d), grad(d))*dx   ))
    E_ela2  = assemble_scalar(form(   0.5 * inner( div(d), div(d))*dx     ))
    if args.dim == 3:
        E_ela3  = assemble_scalar(form(   0.5 * inner( curl(d), curl(d))*dx                             ))
        E_ela4  = assemble_scalar(form(   0.5 *  inner( d, curl(d)) *  inner(d, curl(d))*dx             ))
        E_ela5  = assemble_scalar(form(   0.5 *  inner( cross( d, curl(d)) , cross(d, curl(d))  )*dx    ))

    if H is not None:
        if args.chi_vert != 0.0:
            E_H_vert = assemble_scalar(form(   0.5 * inner( d, H)*inner( d, H)*dx                         ))
        if args.chi_perp != 0.0:
            E_H_perp = assemble_scalar(form(   0.5 * inner(  H, a_times_b_times_c(d, d, H) )*dx           ))
    
    E_ela1      = comm.allreduce(E_ela1, op=MPI.SUM)
    E_ela2      = comm.allreduce(E_ela2, op=MPI.SUM)
    if args.dim == 3:
        E_ela3      = comm.allreduce(E_ela3, op=MPI.SUM)
        E_ela4      = comm.allreduce(E_ela4, op=MPI.SUM)
        E_ela5      = comm.allreduce(E_ela5, op=MPI.SUM)

    if H is not None:
        if args.chi_vert != 0.0:
            E_H_vert = comm.allreduce(E_H_vert, op=MPI.SUM)
        if args.chi_perp != 0.0:
            E_H_perp = comm.allreduce(E_H_perp, op=MPI.SUM)
    
    if args.dim == 3:
        E_ela   = args.K1 * E_ela1 +  args.K2 * E_ela2 + args.K3 * E_ela3 + args.K4 * E_ela4 + args.K5 * E_ela5
    else:
        E_ela   = args.K1 * E_ela1 +  args.K2 * E_ela2 
    E_total = E_ela
    
    if H is not None:
        if args.chi_vert != 0.0:
            E_total -= args.chi_vert * E_H_vert 
        if args.chi_perp != 0.0:
            E_total -= args.chi_perp * E_H_perp 
    
    # DISSIPATION
    dt = args.dt
    dissipation_form =(-1)*dt* args.gamma *inner(q, a_times_b_times_c(d0,d0,q))*dxL

    dissipation = assemble_scalar(form( dissipation_form ))
    dissipation = comm.allreduce(dissipation, op=MPI.SUM)

    # NODAL UNIT-NORM AND ORTHOGONALITY
    d.x.scatter_forward()
    d0.x.scatter_forward()
    orthogonality = np.max(np.abs( np.sum( (np.reshape( d.x.array[:] , (-1, args.dim)) - np.reshape( d0.x.array[:] , (-1, args.dim))) * np.reshape( d0.x.array[:] , (-1, args.dim)) , axis=1 ) ))
    orthogonality = comm.allreduce(orthogonality, op=MPI.MAX)
    unit1 = np.max(np.linalg.norm(np.reshape( d.x.array[:] , (-1, args.dim)), axis=1))    
    unit2 = np.min(np.linalg.norm(np.reshape( d.x.array[:] , (-1, args.dim)), axis=1))
    unit1 = comm.allreduce(unit1, op=MPI.MAX)
    unit2 = comm.allreduce(unit2, op=MPI.MIN)


    res =  {
        "Etot"+id  : E_total,
        "Eela"+id  : E_ela,
        "Eela1"+id : E_ela1,
        "Eela2"+id : E_ela2,
        "diss"+id  : dissipation,
        "orth"+id  : orthogonality,
        "unit1"+id  : unit1,
        "unit2"+id  : unit2
        }

    if args.dim ==3:
        res = res | { "Eela3"+id : E_ela3, "Eela4"+id : E_ela4, "Eela5"+id : E_ela5 }
    if H is not None:
        if args.chi_vert != 0.0:
            res = res | {"E_H_vert" :E_H_vert}
        if args.chi_perp != 0.0:
            res = res | {"E_H_perp" :E_H_perp}
    
    return res




