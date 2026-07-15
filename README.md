# stoch_heat_flow_into_sphere
FEniCSx implementation of some numerical methods for the stochastic heat flow into the sphere.

## Governing Equations
The governing equations in a general form can be stated as:

$$
\begin{aligned}
\mathrm{d} \boldsymbol{d} + \gamma  (\boldsymbol{\mathrm{I}} - \boldsymbol{d} \otimes \boldsymbol{d})  \boldsymbol{q}  \mathrm{d}t & = c_S \Phi (\boldsymbol{d}) \times \circ \mathrm{d} \boldsymbol{\mathrm{W}} , \\
\vert \boldsymbol{d} \vert & = 1 ,
\end{aligned}
$$
where $\boldsymbol{\mathrm{W}}$ is a cylindrical Wiener process. The map $\Phi$ must be represented in form of a weighted Hilbert-Schmidt Operator $\boldsymbol{\mathrm{Q}}^{1/2} $, i.e.
$$
\Phi (\boldsymbol{d}) \times \circ \mathrm{d} \boldsymbol{\mathrm{W}}  = \boldsymbol{d} \times \circ \mathrm{d} (\boldsymbol{\mathrm{Q}}^{1/2}  \boldsymbol{\mathrm{W}} ).
$$


The governing equations are equipped with an elastic energy term, as well as a coupling with a magnetic field,

$$
E = E_{\mathrm{ela}} + E_{ \boldsymbol{H}},
\qquad E_{ \boldsymbol{H} } = - \frac{\chi_{\perp}}{2} \vert  \boldsymbol{d} \times  \boldsymbol{H} \vert^2 - \frac{\chi_{\Vert}}{2} ( \boldsymbol{d} \cdot  \boldsymbol{H})^2 \, .
$$

As elastic energy, we consider the Oseen-Frank energy given by

$$
E_{\mathrm{ela}} = \int_{\Omega} \frac{{K}_1}{2} \vert \nabla  \boldsymbol{d} \vert ^2 + \frac{{K}_2}{2} (\nabla \cdot   \boldsymbol{d})^2 + \frac{{K}_3}{2} \vert \nabla \times  \boldsymbol{d} \vert ^2 + \frac{{K}_4}{2} ( \boldsymbol{d} \cdot \nabla \times  \boldsymbol{d})^2 + \frac{{K}_5}{2} \vert  \boldsymbol{d} \times \nabla \times  \boldsymbol{d} \vert ^2 \mathrm{dx} .
$$

Accordingly, the variational derivative $q$ is defined by

$$
\begin{aligned}
\int_{\Omega} \boldsymbol{q} \cdot \phi \mathrm{dx} \mathrm{d} \tau = & \int_0^T \int_{\Omega} {K}_1 \nabla  \boldsymbol{d} : \nabla \phi + {K}_2 (\nabla \cdot   \boldsymbol{d}) (\nabla \cdot   \phi) + {K}_3 (\nabla \times  \boldsymbol{d}) \cdot (\nabla \times  \phi) \mathrm{dx} \\
& + \int_{\Omega} {K}_4 ( \boldsymbol{d} \cdot \nabla \times  \boldsymbol{d}) ( \boldsymbol{d} \cdot \nabla \times  \phi + \phi \cdot \nabla \times  \boldsymbol{d}) \mathrm{dx} \\ 
& + \int_{\Omega} {K}_5 ( \boldsymbol{d} \times \nabla \times  \boldsymbol{d} ) \cdot ( \phi \times \nabla \times  \boldsymbol{d}  +  \boldsymbol{d} \times \nabla \times \phi) \mathrm{dx} .
\end{aligned}
$$

In the case of the Dirichlet energy, 

$$
E = E_{\mathrm{ela}} = \int_{\Omega} \frac{1}{2} \vert \nabla  \boldsymbol{d} \vert ^2 ,
$$


the variational derivative can be simplified,

$$
\int_{\Omega} \boldsymbol{q} \cdot \phi \mathrm{dx} \mathrm{d} \tau = 
\int_0^T \int_{\Omega} \nabla  \boldsymbol{d} : \nabla \phi \mathrm{dx},
\qquad \implies \qquad \boldsymbol{q} = - \Delta \boldsymbol{d} ,
$$

and so the partial differential equation becomes,

$$
\begin{aligned}
\mathrm{d} \boldsymbol{d} - \gamma  (\boldsymbol{\mathrm{I}} - \boldsymbol{d} \otimes \boldsymbol{d})  \Delta \boldsymbol{d}  \mathrm{d}t & = c_S \Phi (\boldsymbol{d}) \times \circ \mathrm{d} \boldsymbol{\mathrm{W}} , \\
\vert \boldsymbol{d} \vert & = 1 ,
\end{aligned}
$$

Equivalently this is often written as

$$
\begin{aligned}
\mathrm{d} \boldsymbol{d} + \gamma  \boldsymbol{d} \times (\boldsymbol{d} \times \Delta \boldsymbol{d})  \mathrm{d}t & = c_S \Phi (\boldsymbol{d}) \times \circ \mathrm{d} \boldsymbol{\mathrm{W}} , \\
\vert \boldsymbol{d} \vert & = 1 .
\end{aligned}
$$

## Approximation of the Wiener process
The vector-valued Wiener process is decomposed into 
$$
\boldsymbol{\mathrm{W}}^i (t) = \sum_{l \in \mathbb{Z}} \boldsymbol{\mathrm{g}}_{l,i} \boldsymbol{e}_i \beta^i_l (t),
$$
with $\{\beta^i_l \}_{i=1,..,d}$ being independent Brownian Motions and $\boldsymbol{\mathrm{g}}_{l,i}$ being an orthonormal base for the space $L^2(\Omega; \mathbb{R}^d)$.
In particular, we choose a Fourier Decomposition on the unit domain $(-0.5,0.5)^d$ by
$$
\boldsymbol{\mathrm{g}}_{l,i}
=
\boldsymbol{e}_i
\begin{cases}
1
\text{ if } l = 0 \,,\\
\sqrt{2}\sin (2 \pi l x_i)
\text{ if } l > 0 \,,\\
\sqrt{2}\cos (2 \pi l x_i)
\text{ if } l < 0 \,.
\end{cases}
$$
Using the same function base, the Hilbert--Schmidt operator $\boldsymbol{\mathrm{Q}}^{1/2}$ can be represented by coefficients $\{\lambda_{l,i}\}_{l,i}$
$$
\boldsymbol{\mathrm{Q}}^{1/2} \boldsymbol{g} = \sum_{i=1}^d \sum_{l \in \mathbb{Z}} \lambda_{l,i} (\boldsymbol{g},\boldsymbol{\mathrm{g}}_{l,i})_{L^2} \boldsymbol{\mathrm{g}}_{l,i} \, .
$$
Several choices for $\lambda$ are available via the Command-Line-Interface via:
```
python -m sim.run -lam i
```
- $i=1$ corresponds to setting $\lambda_{l,i} =1$
- $i=2$ corresponds to setting $\lambda_{l,i} =2^{-l}$
- $i=3$ corresponds to setting $\lambda_{l,i} =(2l^2)^{-l}$

As a consequence, the Stratonovich integral,
$$
\int^{t^{n+1}}_{t^n} \boldsymbol{d} \times \circ \mathrm{d} (\boldsymbol{\mathrm{Q}}^{1/2}  \boldsymbol{\mathrm{W}} )  \, ,
$$
can be approximated by considering a finite sum,
$$
\int^{t^{n+1}}_{t^n} \boldsymbol{d}^{n+1/2} \times \circ \mathrm{d} (\boldsymbol{\mathrm{Q}}^{1/2}  \boldsymbol{\mathrm{W}} )
=
\boldsymbol{d}^{n+1/2} \times \sum_{i=1}^d \sum_{l \in \mathbb{Z}} \lambda_{l,i} \boldsymbol{\mathrm{g}}_{l,i}
\int^{t^{n+1}}_{t^n} \mathrm{d} \beta^i_l
\approx
\boldsymbol{d}^{n+1/2} \times \sum_{i=1}^d \sum_{l \in \mathbb{Z} \cap [-Z_h, Z_h]} \lambda_{l,i} \boldsymbol{\mathrm{g}}_{l,i}
\int^{t^{n+1}}_{t^n} \mathrm{d} \beta^i_l
\, ,
$$
where $Z_h$ depends on the mesh width. On a cartesian grid of a with $1/h+1$ points in each dimensional direction, the points have the coordinates $[0, 1/h, 2/h, ... , 1]$. Accordingly, at the interpolation points (= mesh nodes), the basis functions with the frequencies $l=1/h$ and $l=2/h$ coincide. As a heuristic, we set $Z_h = 2/h -1$.

## Numerical Methods

Currently the following numerical schemes are available (key : explanation).
- nonlin_cg : Numerical Method in [[1]](#1)[[4]](#4) (setting the velocity to zero in time and space). Implemented using a monolithic Newton solver.
- linear_cg : Algorithm 1 in [[3]](#3)[[4]](#4) (setting the velocity to zero in time and space). Linear projection method.
- linear_dg : Algorithm 2 in [[3]](#3)[[4]](#4) (setting the velocity to zero in time and space). Linear projection method based on the DG method.
- fp_decoupled : Numerical Method in [[2]](#2)[[4]](#4) (setting the velocity to zero in time and space) with an iterative Picard-type linearization, see [[2]](#2).
- fp_coupled : Numerical Method in [[4]](#4) (setting the velocity to zero in time and space) with an iterative Picard-type linearization similar to fp_decoupled.


They can be selected by specifying the command line key "-m" or "--mod".

## Numerical Settings

- spiral : spiral domain with a known stationary solution, see [[3,5]](#5).
- smooth : smooth initial condition on a unit square, see [[4]](#4).
- annihilation : two line defects in a unit-cube, see [[1]](#1).

They can be selected by specifying the command line key "-e" or "--exp".

## Getting Started and Usage

All arguments to run simulations are given via the command line input. To see the options run the following command in the package directory:

```
python -m sim.run -h
```

Another usage example with several arguments:

```
python -m sim.run -m linear_cg -e unstable -vtx 1 -dt 0.01 -sid "experiment1" -T 0.05
```

Presets for several simulations are given in the folder 'sim/sim_presets/' usually in the form of a bash or python file. Examples for usage:
```
sim/sim_presets/unit.sh
```

Physical parameters of the governing equations and the energy term can be changed via command line arguments, i.e.
```
python -m sim.run -m linear_cg -e unstable -gamma 2.0 -K1 1.0 -K2 0.5 -K3 0.0 -K4 0.1 -K5 0.1 -chi_vert -1.0 -chi_perp -0.5
```

Simulations can also be run from an existing config file:
```
python -m sim.runconfig "output/unit1/config.json"
```

## Requirements

All requirements can be found in the file requirements.txt and can be installed via pip by

```
pip install -r requirements.txt
```

or via conda by

```
conda create --name my-env-name --file requirements.txt -c conda-forge
```

## References to relevant publications

<a id="1">[1]</a> 
Lasarzik, R., & Reiter, M. E. V. (2023). Analysis and numerical approximation of energy-variational solutions to the Ericksen-Leslie equations. Acta Appl. Math., 184, 44. https://doi.org/10.1007/s10440-023-00563-9

<a id="2">[2]</a> 
Reiter, M. E. V. (2026). Decoupling and Linearization of a Liquid Crystal Model fulfilling a Unit Norm Constraint. Proceedings of ECMI 2023.

<a id="3">[3]</a> 
Reiter, M. E. V. (2025). Projection Methods in the Context of Nematic Crystal Flow. https://arxiv.org/abs/2502.08571 & https://www.doi.org/10.1093/imanum/drag013 

<a id="4">[4]</a>
Reiter, M. E. V. (2026).  Structure-preserving numerical approximation of the Ericksen–Leslie equations for nematic liquid crystal flow. PhD Thesis. https://doi.org/10.14279/depositonce-26183

## Authors

* **Maximilian E. V. Reiter**, https://orcid.org/0000-0001-9137-7978

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details