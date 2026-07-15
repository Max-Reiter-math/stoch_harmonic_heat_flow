import numpy as np

def stratonovich_noise(x: np.ndarray, dh: int, dt: float, get_lambda: callable, dim = 2)-> np.ndarray:
    """
    Stratonovich noise term for the stochastic heat flow into a sphere:
    d(Q^{1/2})W) with W being a cylindrical Wiener process.

    For simplicity, we set Q = Id.

    dh : amount of subdivions at each axis of the cartesian grid. The total number of points is (dh+1)^d where d is the dimension of the problem.

    For dimension d and n = T/dt time steps consider the quantities:
    Z_h : approximation of the noise term given the mesh width h>0

    \mathcal{g} \in \mathbb{R}^{\mathcal{Z}_h \times d \times d}    (basis of L^2 functions in space satisfying $\mathcal{g}_i,l \cdot e_j = 0$ if $i \neq j$)
    \lambda \in \mathbb{R}^{\mathcal{Z}_h \times d}                 (Gram-Schmidt coefficients of the Operator Q with respect to the basis $\mathcal{g}$ of L^2 functions in space)
    such that $\mathcal{g}_l$ is a diagonal matrix. Accordingly, with a change of variables it suffices to save it in an array of the form $\mathbb{R}^{\mathcal{Z}_h \times d }$
    \beta \in \mathbb{R}^{\mathcal{Z}_h \times d}         (vector-valued brownian motion)
    
    Then the noise term is given by:
    N^{n+1} = \sum_{i=1}^{d} \sum_{l=-Z_h}^{Z_h} \lambda_{l,i} \mathcal{g}_{l,i} \beta_{l,i}^{n+1}

    Accordingly, the second sum has M_h = 2*Z_h + 1 terms and the first sum has d terms. 
    """

    Z_h = 2*dh - 1 # NOTE - that on a cartesian grid of a line with dh+1 points, the points have the coordinates [0, 1/dh, 2/dh, ..., 1]. Accordingly, at the interpolation points (= mesh nodes), the functions with the frequencies dh and 2*dh coincide. As a heuristic, we only consider the frequencies 1, ..., 2*dh-1.
    M_h = 2*Z_h + 1

    # ourput array of shape (dim, points) where points is the number of points in the mesh
    values = np.zeros((dim, x.shape[1])) 

    # NOTE - We need to use a mixture of vectorization and loops due to the high-dimensionality of the problem

    # SUMMATION OVER BASIS FUNCTIONS
    # as basis for $\mathcal{g}$ we choose the Fourier base of L^2 consisting of the functions 1, \sqrt{2}*sin(2*pi*k*x), \sqrt{2}*cos(2*pi*k*x) for k\in {1,...,Z_h}.
    # NOTE - We hereby assume the domain to be a unit square. Otherwise the base would not be orthonormal.

    # Basis function: 1
    d_beta = np.random.normal(loc=0.0, scale=dt, size=None) # difference of time partitioned vector-valued brownian motion    
    values[0] += get_lambda(0) * 1.0 * d_beta
    values[1] += get_lambda(0) * 1.0 * d_beta

    for l in range(1,Z_h+1,1):
        frequ = 2*np.pi*l*1.0 # frequency of the basis function

        # Basis function: \sqrt{2}*sin(2*pi*k*x)
        d_beta = np.random.normal(loc=0.0, scale=dt, size=None) # difference of time partitioned vector-valued brownian motion    
        values[0] += get_lambda(l) * 2**(1/2)*np.sin(frequ*x[0]) * d_beta
        values[1] += get_lambda(l) * 2**(1/2)*np.sin(frequ*x[1]) * d_beta

        # Basis function: \sqrt{2}*cos(2*pi*k*x)
        d_beta = np.random.normal(loc=0.0, scale=dt, size=None) # difference of time partitioned vector-valued brownian motion    
        values[0] += get_lambda(l) * 2**(1/2)*np.cos(frequ*x[0]) * d_beta
        values[1] += get_lambda(l) * 2**(1/2)*np.cos(frequ*x[1]) * d_beta
    
    return values


def get_lambda_method(method: int):
    if method == 1:
        def get_lambda(l):
            lam = 1.0       # Setting Lambda to 1.0 as Q is set to the identity map.
            return lam
    elif method == 2:
        def get_lambda(l):
            lam = 2**(-l)   # This defines an Operator Q implicitly and also fulfills the boundedness of the according series.
            return lam
    elif method == 3:
        def get_lambda(l):
            lam = (2*l*l)**(-l)   # This defines an Operator Q implicitly and also fulfills the boundedness of the according series.
            return lam

    return get_lambda