# Higher order stray field computation via Super-Potential Method

Magnetostatic stray field computation is a core component in micromagnetic
simulations. This library implements high-order stray field computation utilizing 
the super-potential method. Both the magnetization and the resulting magnetic field 
are represented using higher-order B-spline bases. The super-potential, 
which circumvents the need to convolve with a singular kernel, is efficiently
approximated with Gaussian sums, leading to a separable expansions on the tensor 
product domain. The field is then represented with high accuracy in functional 
Tucker tensor format, which allows for continuous field evaluation.


## Features


- Computation of the super-potential and Newton-potential on tensor-product domains
- B-spline functional tucker tensors for very accurate function approximation on tensor-product domains
- Fast and accurate micromagnetic field and energy computation using Gaussian sums
- Support for multi tensor product domains
- Full `jit` compilable and GPU friendly *JAX* implementation

## Installation

```bash
git clone https://github.com/schaffer9/ho-stray-field
cd ho-stray-field
pip install -e .
```

## Usage
As a quick example, we compute the micromagnetic energy of a flower state:

```python
import jax
import jax.numpy as jnp
from ho_stray_field import (
    TensorGrid,
    BSpline,
    PotentialState, 
    solve_energy, 
    fit_mag,
    superpotential,
    scalar_potential,
    stray_field,
    energy,
    solve_energy
)
from ho_stray_field.sources import flower_state
jax.config.update("jax_enable_x64", True)  # double precision is required

# first we define the B-spline model on the tensor product domain, where we use 
# a multilinear rank of 40
model = BSpline(TensorGrid(*[jnp.linspace(-0.5, 0.5, 40)] * 3), degree=7)

# Then we create a solver object which already performs necessary precomputations
# such as integration of GS integrals for the target points.
# Note that the target points are also given on a tensor grid and we use
# 200 Gauss–Legendre quadrature points for each axis.
quad_grid = TensorGrid(*[jnp.linspace(-0.5, 0.5, 2)] * 3).to_gauss(200)
solver = PotentialState.init(
    pot_elm=model,
    target_quad_grid=quad_grid,
)

# The solver can then be used to fit a magnetization and compute the respective
# potential or fields. If not defined otherwise, the same model is used to
# fit magnetization and potential.
# We can fit the flower state into the functional Tucker format:
mag = fit_mag(solver, flower_state)

# Once this is done we can compute the superpotential,
sp = superpotential(solver, mag)
# the scalar potential,
phi = scalar_potential(solver, sp)
# the stray field
h = stray_field(solver, phi)
# and the energy
energy(h, mag, quad_grid)

# If only the energy is required we can compute it directly with
solve_energy(solver, mag)
```

More examples can be found in the **examples** folder.


## Acknowledgements
Financial support by the Austrian Science Fund (FWF) via project ”Data-driven Reduced Order Approaches for Micromagnetism (Data-ROAM)” (Grant-DOI: 10.55776/PAT7615923) and project ”Design of Nanocomposite Magnets by Machine Learning (DeNaMML)” (Grant-DOI: 10.55776/P35413) is gratefully acknowledged. The authors acknowledge the University of Vienna research platform MMM Mathematics - Magnetism - Materials. The computations were partly achieved by using the Vienna Scientific Cluster (VSC) via the funded projects No. 71140 and 71952. This research was funded in whole or in part by the Austrian Science Fund (FWF) [10.55776/PAT7615923, 10.55776/P35413].

