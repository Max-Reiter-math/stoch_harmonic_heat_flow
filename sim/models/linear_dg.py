from functools import partial
from ufl import grad, div, inner, dot, avg, jump, dx, FacetNormal, CellDiameter, Measure, TrialFunction, TestFunction
import dolfinx.la as la
from dolfinx.fem import Function, functionspace, form, assemble_scalar, ElementMetaData, bcs_by_block, extract_function_spaces, bcs_by_block
from dolfinx.fem.petsc import assemble_matrix, assemble_matrix_nest, assemble_vector_nest, apply_lifting_nest, set_bc_nest, create_vector_nest, set_bc_nest, LinearProblem
from sim.common.grad_dg0 import discrete_gradient_def, d_to_grad_d_mappings, reconstruct_grad, interpolate_dg0_at, compute_dg0_int_pts_on_bdry
from sim.common.operators import *
from sim.common.common_fem_methods import *
from sim.common.meta_bcs import *
from sim.common.stochastics import stratonovich_noise, get_lambda_method

def linear_dg(comm, experiment, args, postprocess=None):
    """
    Algorithm 2 (setting the velocity to zero in time and space) by Maximilian E. V. Reiter. (2025). Projection Methods in the Context of Nematic Crystal Flow.

    Important properties of the algorithm:
    - fulfills a discrete energy law
    - unconditional existence
    - automatically fulfills unit-norm constraint
    - assumes restrictive mesh-conditions
    """

    if postprocess and comm.rank == 0:
        postprocess.log("dict", "static", {"# MPI Ranks": comm.size})
        
    # SECTION PARAMETERS
    dim = args.dim
    dt  = args.dt
    t   = 0
    
    if postprocess and comm.rank == 0:
        postprocess.log("dict", "static",{"SID": args.sim_id, "MODEL":args.mod, "PROJECTION STEP": args.projection_step, "PROJECT TANGENT MATRIX": args.project_tangent_map, "ALPHA" : args.alpha, "EXPERIMENT":args.exp, "mesh res.": args.dh, "dt":args.dt, "dim":args.dim, "T": args.T, "seed":args.seed, "cs":args.cs, "approx_method": args.lam})

    #!SECTION PARAMETERS
    domain          = experiment.mesh    
    meshtags        = experiment.meshtags
    dg0_cells       = experiment.dg0_cells
    dg0_int_points  = experiment.dg0_int_points
    dx          = Measure('dx', domain=domain) 
    ds          = Measure('ds', domain=domain, subdomain_data=meshtags) # since we allow for mix of boundary conditions                                         
    dS          = Measure('dS', domain=domain)
    n_F         = FacetNormal(domain)
    h_T         = CellDiameter(domain)

    #add all local variables for transparency    
    if postprocess and comm.rank == 0:
        postprocess.log("dict","static",{"model.vars":dict(locals())}, visible =False)

    initial_conditions          = experiment.initial_conditions
    boundary_conditions         = experiment.boundary_conditions

    total_time_start = mpi_time(comm)
    
    # SECTION - FUNCTION SPACES AND FUNCTIONS

    P2          = functionspace(domain, ElementMetaData("Lagrange", 2 , shape=(dim,))) 
    P1          = functionspace(domain, ("Lagrange", 1) )
    D, Y        = functionspace(domain, ElementMetaData("DG", 0 , shape=(dim,))), functionspace(domain, ElementMetaData("DG", 0 , shape=(dim,)))
    TensorF     = functionspace(domain, ElementMetaData("DG", 0 , shape=(dim, dim))) # function space for the reconstructed gradient
    
    d1, grad_d1, q1                  = TrialFunction(D),TrialFunction(TensorF), TrialFunction(Y)
    d0, grad_d0, q0                  = Function(D), Function(TensorF), Function(Y)
    d_                                      = Function(D) # d_ is used for the tangent projection
    d_bc                                    = Function(D) # d_bc is used for the weak anchoring of the dirichlet boundary condition
    d, grad_d, q                       = Function(D), Function(TensorF), Function(Y)
    c_test, tau, b_test     = TestFunction(D), TestFunction(TensorF), TestFunction(Y)      

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
    local_info = {f"rank {comm.rank} dofs": local_dofs}                                                                                             # Create local dictionary   
    all_info = comm.gather(local_info, root=0)                                                                                                      # Gather all dictionaries at root

    if comm.rank == 0:
        # Merge all into a single dict
        combined = {}
        for partial_dict in all_info:
            combined.update(partial_dict)
        postprocess.log("dict", "static", combined)

    #!SECTION FUNCTION SPACES AND FUNCTIONS
    
    

    #SECTION - BOUNDARY CONDITIONS
    bcs = []
    
    for bcwofs in boundary_conditions:
        if bcwofs.type == "Dirichlet":
            if bcwofs.type == "Dirichlet" and bcwofs.quantity == "d":
                #FIXME - wrong interpolation points!
                b_cells, b_int_pts = compute_dg0_int_pts_on_bdry(domain, dg0_cells, dg0_int_points, marker = bcwofs.marker)
                interpolate_dg0_at(d_bc, bcwofs.values, b_cells, b_int_pts) # DG version of interpolation
        # elif bcwofs.type == "Neumann":            
        # elif bcwofs.type == "Robin":
        else: postprocess.log("dict", "static",{"Warning" : "Boundary conditions of type "+bcwofs.type+" are currently not implemented and will be ignored..."} )
    
    #!SECTION

    # SECTION VARIATONAL FORMULATION
    a, L = variational_form(D, TensorF, d1, q1, grad_d1, d0, q0, grad_d0, b_test, c_test, tau, d_, d_bc, dt, n_F, h_T, args, ds, dS, H=None, noise = None,  boundary_conditions=boundary_conditions, postprocess = postprocess, dg0_cells = dg0_cells, dg0_int_points = dg0_int_points)
    
    # SECTION INITIAL CONDITIONS
    interpolate_dg0_at(d0, initial_conditions["d"], dg0_cells, dg0_int_points) # DG version of interpolation
    interpolate_dg0_at(d_, initial_conditions["d"], dg0_cells, dg0_int_points) # DG version of interpolation
    scatter_all([d0, d_, d_bc])
    # Create Matrix and Vector for the stationary Equation System to reconstruct the gradient
    B, res = d_to_grad_d_mappings(TensorF, grad_d1, tau, D , d1, boundary_conditions, n_F, ds,dS, cells = dg0_cells, int_points = dg0_int_points )
    # Compute reconstructed gradient for t = 0
    reconstruct_grad(B,res, grad_d0, d0) # NOTE - the result is saved to the function grad_d0
    
    # COMPUTATION OF INITIAL DIVERGENCE (BECAUSE WE USE IT TO SET THE BOUNDARY CONDITIONS FOR q)
    # d0 and grad_d0 already exist, it suffices to compute q from that accordingly
    #TODO - add function
    a_23, a_21, L_2 = discrete_laplacian_def(grad_d0, d0, b_test, d_bc, n_F, args.alpha, h_T, boundary_conditions, ds, dS)
    problem0 = LinearProblem(form(inner(qc1, bc_test)*dx), form(L_2 - a_23 - a_21),  bcs=[], u=q_ic, petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"})
    problem0.solve()
    q0.interpolate(q_ic)
    q0.x.scatter_forward()
    
    #!SECTION

    #SECTION - POSTPROCESSING FOR t=0
    DG1  = functionspace(domain, ElementMetaData("DG", 1 , shape=(dim,))) # space used to output to vtx
    DG1T = functionspace(domain, ElementMetaData("DG", 1 , shape=(dim,dim)))
    d_out, q_out, dbc_out = Function(DG1), Function(DG1), Function(DG1) # vtx cant deal with dg0, but with dg1
    grad_d_out = Function(DG1T)
    # NOTE - here the classical interpolation is sufficient since we simply save DG0 functions in a DG1 space. The interpolation is therefore exact.
    d_out.interpolate(d0)
    q_out.interpolate(q0)
    dbc_out.interpolate(d_bc)
    grad_d_out.interpolate(grad_d0)
    scatter_all([d_out, q_out, grad_d_out,dbc_out])
    if postprocess:
        postprocess.log_functions(0.0, {"d":d_out, "grad_d": grad_d_out, "q":q_out , "dbc": dbc_out, "n": noise}, mesh = domain) #, meshtags = meshtags)

    metrics = compute_metrics(comm, args, d0,grad_d0, d0,q0, d_bc, h_T, ds, dS, id ="", postprocess = postprocess) # for initial condition
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
        A, b = assemble_all(a, L, B, res, bcs = bcs)
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
        d.x.scatter_forward()
        # Compute reconstructed DG= gradient afterwards
        reconstruct_grad(B,res, grad_d, d)
        
        time_sol = mpi_time(comm, start = start_sol)

        #!SECTION

        #SECTION - EVTL. METRICS BEFORE PROJECTION STEP    
        if args.projection_step ==1:
            metrics = compute_metrics(comm, args, d,grad_d, d0,q, d_bc, h_T, ds, dS, id=".b4p", postprocess = postprocess)
            if postprocess and comm.rank == 0:
                postprocess.log("dict", t, { "time" : t} | metrics , visible = False)
                postprocess.log("dict", t, {"time":t, "t.ass": assembly_time, "t.sol": time_sol, "t.solsetup" : time_solsetup}, visible = True)

        #!SECTION

        # SECTION - NODAL PROJECTION STEP
        if args.projection_step ==1:
            start_pstep = mpi_time(comm)

            nodal_normalization(d, dim)

            time_pstep = mpi_time(comm, start = start_pstep )
        #!SECTION 

        #SECTION - UPDATE
        update_and_scatter([d0,d_,q0, grad_d0], [d,d,q,grad_d])
        reconstruct_grad(B,res, grad_d0, d0)
        #!SECTION 

        # SECTION - NODAL PROJECTION STEP FOR TANGENT MAP
        if args.project_tangent_map == 1:
            start_pstep2 = mpi_time(comm)

            nodal_normalization(d, dim)

            time_pstep2 = mpi_time(comm, start = start_pstep2 )
        #!SECTION 

        
        
        #SECTION - METRICS AT END OF ITERATION 
        metrics =  compute_metrics(comm, args, d,grad_d, d0,q, d_bc, h_T, ds, dS, postprocess = postprocess)

        errorL2 = np.nan
        if experiment.has_exact_solution:   
            errorL2 = experiment.compute_error(comm, d,t,norm = "L2", degree_raise = 3, family = "DG")   

        total_time = mpi_time(comm, start = total_time_start )
        if postprocess and comm.rank == 0:
            postprocess.log("dict", t, { "time" : t, "errorL2" : errorL2 , "t.tot" : total_time} | metrics)
            if args.projection_step ==1: postprocess.log("dict", t, { "time" : t, "t.pstep" : time_pstep})
            if args.project_tangent_map == 1: postprocess.log("dict", t, { "time" : t, "t.pstep2" : time_pstep2})
        
        #!SECTION

        #SECTION - SAVING
        d_out.interpolate(d0)
        q_out.interpolate(q0)
        dbc_out.interpolate(d_bc)
        grad_d_out.interpolate(grad_d0)
        scatter_all([d_out, q_out, grad_d_out,dbc_out])
        if postprocess: 
            postprocess.log_functions(t, {"d":d_out, "grad_d": grad_d_out, "q":q_out, "dbc": dbc_out, "n": noise}, mesh = domain, meshtags = meshtags)

        #!SECTION

    #!SECTION TIME EVOLUTION

    if postprocess: 
        postprocess.close()

#!SECTION GENERAL METHOD



def variational_form(DFS, TFS, d1, q1, grad_d1, d0, q0, grad_d0, b, c, tau, d_, d_bc, dt, normal_F, h_T, args, ds, dS, H=None, noise=None, boundary_conditions=[], postprocess = None, dg0_cells: np.ndarray = None, dg0_int_points: np.ndarray = None):
        """
        The variational form is structured as followed:
        a(.,.) : bilinear form describing the lhs of the system
        L(.) : linear form describing the rhs of the system
        a_ij(.,.) : bilinear form describing the lhs of the system depending on the i-th test function and the j-th trial function
        """
        # SECTION DIRECTOR EQUATION
        a_11 = inner(d1, c)*dx

        if args.gamma != 0.0:
            a_12 = (-1)*dt* args.gamma *inner(q1, a_times_b_times_c(d_,d_,c))*dx

        L_1 = inner(d0, c)*dx

        if not( noise is None ):
            if args.dim == 2:
                d_orth = as_vector([-d_[1], d_[0]])
                L_1 += args.cs*inner(noise,d_)*inner(d_orth,c)*dx
            else:
                L_1 += args.cs*inner(cross(d_, noise),c)*dx

        #!SECTION DIRECTOR EQUATION

        # SECTION EQUATION FOR THE VARIATIONAL DERIVATIVE
        # mass matrix for q
        a_22 = (-1)* inner(q1, b)*dx 

        a_23, a_21, L_2 = discrete_laplacian_def(grad_d1, d1, b, d_bc, normal_F, args.alpha, h_T, boundary_conditions, ds, dS)
        
        # evtl. add energy parts that depend on the magnetic field
        if H != None and args.chi_vert != 0.0:
            a_21 -= args.chi_vert * inner( d1, H)*inner( b, H)*dx
        if H != None and args.chi_perp != 0.0:
            a_21 += args.chi_perp * inner(  H, a_times_b_times_c(d1, b, H) )*dx

        """
        RECONSTRUCTED GRADIENT (All terms dependent on test function tau)
        """
        
        a_33 = (-1)*args.K1 * inner( grad_d1, tau)*dx

        if args.K2 != 0.0 or args.K3 != 0.0 or args.K4 != 0.0 or args.K5 != 0.0:
            if postprocess:
                postprocess.log_message("Values other than 0 for K2,K3,K4,K5 are not supported for this model")
        
        # NOTE - the following forms are only filled to better understand the full system. At this point they are not necessary for the variational formulation.
        a_31, L_3 = None, None
        # a_31, L_3 = discrete_gradient_def(grad_d_FS = TFS, tau_test= tau, d_FS = DFS, d_trial = d1, boundary_conditions=boundary_conditions, normal_F=normal_F,  ds=ds, dS=dS, cells = dg0_cells, int_points = dg0_int_points)


        #!SECTION EQUATION FOR THE VARIATIONAL DERIVATIVE
        
        """
        Full bilinear form (lhs) and linear form (rhs)
        """
        a = [
            [a_11, a_12, None],
            [a_21, a_22, a_23],
            [a_31, None, a_33],
        ]

        L = [
            L_1 ,
            L_2 ,
            L_3 ,
        ] 

        return a, L



def discrete_laplacian_def(grad_d_trial, d_trial, b_test, d_bc, normal_F, alpha, h_T, boundary_conditions, ds, dS ):
    """
    Variational Formulation of the RHS of the discrete Laplacian, i.e.
        inner(q,b_test)*dxL
        =
        a_23 + a_21 - L4
    """
    # NOTE - ds needs to be given as argument due to the meshtags being initialized prior
    # NOTE - no parallel architecture needed here, since we only initialize a variational formulation

    # applying the Definition of the discrete lifting onto the discrete gradient of the test function b
    # NOTE - using the normal in direction '-' is consistent with the Definition of the discrete gradient and is necessary for the right results!
    a_23 = inner(dot(avg(grad_d_trial),normal_F('-')),jump(b_test))*dS      # on the interior

    # Penalization terms
    a_21 = (alpha/avg(h_T)) * inner( jump(d_trial), jump(b_test) )*dS   # interior jump penalization 

    # initialize the rhs in this row with zero
    zero_vec = Function(d_bc.function_space)
    L_2 = inner(zero_vec, b_test)*dx

    # boundary condition
    for bcwofs in boundary_conditions:
        if bcwofs.type == "Dirichlet" and bcwofs.quantity == "d":
            a_23 += -inner(dot(grad_d_trial, normal_F), b_test)*ds(bcwofs.meshtag)             # Definition of the discrete lifting on the boundary        
            a_21 += (alpha/h_T)*inner(d_trial, b_test)*ds(bcwofs.meshtag)     # bc penalization
            L_2 += (alpha/h_T)*inner( d_bc, b_test)*ds(bcwofs.meshtag)     # bc penalization
    
    return a_23, a_21, L_2


def setup_split_solver(comm, args,A):
    # Create a nested matrix P to use as the preconditioner. 
    P = PETSc.Mat().createNest([
        [A.getNestSubMatrix(0, 0), None],
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
    ksp.getPC().setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE) # NOTE - this time we use additive since it is better in parallel

    # Return the index sets representing the row and column spaces. 2 times Blocks spaces
    nested_IS = P.getNestISs()
    ksp.getPC().setFieldSplitIS(("d", nested_IS[0][0]), ("q", nested_IS[0][1]))

    # Set the preconditioners for each block via CLI.
    ksp_d, ksp_q = ksp.getPC().getFieldSplitSubKSP()
    ksp_d.setType(args.ksp_type_d) 
    ksp_d.getPC().setType(args.pc_type_d) 
    ksp_q.setType(args.ksp_type_q) 
    ksp_q.getPC().setType(args.pc_type_q) 

    return ksp

def assemble_all(a, L, B, res, bcs = []):
    """
    Recall that
    a = [   [a_11, a_12, None],
            [a_21, a_22, a_23],
            [a_31, None, a_33] ]
    L = [L_1 , L_2 , L_3 ] 
    """
    a_reduced = form([[a[i][j] for j in range(2)] for i in range(2)])
    L_reduced = form([L[i] for i in range(2)])
    # Assemble nested matrix operators
    A = assemble_matrix_nest(a_reduced, bcs=bcs)
    A.assemble()        
    b = assemble_vector_nest(L_reduced) 
    apply_lifting_nest(b, a_reduced, bcs=bcs)

    # Ghost update
    b_sub_vecs = b.getNestSubVecs()
    for b_sub in b_sub_vecs:
        b_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    # Set Dirichlet boundary condition values in the RHS vector
    bcs0 = bcs_by_block(extract_function_spaces(L_reduced), bcs)
    set_bc_nest(b, bcs0)

    
    a_23 = form(a[1][2])
    """
    The last row can be ignored, since we alredy initialized the map from d.x to grad_d.x in the beginning with the method: 
        B, res = d_to_grad_d_mappings(TensorF, grad_d1, tau, d1, d_bc, n_F, boundary_conditions = [])

    Accordingly, we do not invoke:
        a_31 = form(a[2][0])
        a_33 = form(a[2][2])
    """   
    A_23 = assemble_matrix(a_23, bcs=bcs)
    A_23.assemble()
    """
    Recall, how matrix B and vector res map from d.x to grad_d.x:
    grad_d.x = B d.x + res
    """
    A_23res = b.getNestSubVecs()[1].copy() # get nested vector
    res.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    A_23.mult(res, A_23res)  # multiply A_23res = A45 res
    b_sub_vecs[1].axpy(-1.0, A_23res) # b4 <- b4 - A_23*res
    b_sub_vecs[1].ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    b.assemble()
    
    A_23B = A_23.matMult(B) # A_23 <- A_23 *B

    A_21 = A.getNestSubMatrix(1, 0)
    A_21.axpy(1.0, A_23B) #, structure=PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN)
    A.assemble() # since matrix was manually modified, we need to reassemble

    return A,b

def compute_metrics(comm, args, d,grad_d, d0,q, dbc, h_T, ds, dS,  id ="", postprocess = None):
    # ENERGY TERMS
    E_Ji    = assemble_scalar(form(   0.5 * (1/avg(h_T)) * inner( jump(d), jump(d) )*dS       ))
    E_Jbc   = assemble_scalar(form(   0.5 * (1/h_T)      * inner( d-dbc, d-dbc )*ds           ))
    E_ela1  = assemble_scalar(form(   0.5 *                inner(grad_d, grad_d)*dx           ))

    if args.K2 != 0.0 or args.K3 != 0.0 or args.K4 != 0.0 or args.K5 != 0.0:
        if postprocess:
            postprocess.log_message("Values other than 0 for K2,K3,K4,K5 are not supported for this model")
    
    E_Ji        = comm.allreduce(E_Ji, op=MPI.SUM)
    E_Jbc       = comm.allreduce(E_Jbc, op=MPI.SUM)
    E_ela1      = comm.allreduce(E_ela1, op=MPI.SUM)
    E_ela   = args.K1 * E_ela1 + args.alpha*E_Ji + args.alpha*E_Jbc
    E_total = E_ela
    
    #TODO - Magnetic field

    # DISSIPATION
    dt = args.dt
    dissipation_form =(-1)*dt* args.gamma *inner(q, a_times_b_times_c(d0,d0,q))*dx

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
        "EJi"+id    : E_Ji,
        "EJbc"+id  : E_Jbc,
        "Eela1"+id : E_ela1,
        "diss"+id  : dissipation,
        "orth"+id  : orthogonality,
        "unit1"+id  : unit1,
        "unit2"+id  : unit2
        }
    
    return res




