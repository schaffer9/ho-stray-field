import timeit
from functools import partial

import pytest
from quadax import quadgk

from tpelm.bspline import BSpline
from tpelm.tensor_grid import TensorGrid
from tpelm.functional_tucker import fit, fit_divergence, fit_laplace, fit_grad
from tpelm.tucker_tensor import TuckerTensor, tucker_dot
from tpelm.gs import superpotential, fit_superpotential, superpotential_factors, merge_quad_info, GS
from tpelm.magnetostatic import StrayFieldSolver

from .. import *
from ..sources import flower_state, vortex_state, m_uniform
from .utils import write_csv_row


@pytest.mark.functional
class TestEnergy:
    quadrature_points = 140

    def setup_class(self):
        jax.config.update("jax_enable_x64", True)

    def teardown_class(self):
        jax.config.update("jax_enable_x64", False)
        


    def test_energy_uniform_cube(self, artifact_dir):
        
        csv_file = artifact_dir / "sp_uniform.csv"
        
        #@partial(jax.jit, device=jax.devices("cpu")[0])
        def _energy():
            degree = 6
            n = 39
            tg_m = TensorGrid(*([jnp.linspace(-0.5, 0.5, 2)] * 3))
            elm_m = BSpline(tg_m, degree=0)
            tg_quad = tg_m.to_gauss(3)
            inv_factors = elm_m.pinv(tg_quad)
            # F = jnp.apply_along_axis(m_uniform, -1, tg_quad.grid)
            F = jnp.apply_along_axis(m_uniform, -1, tg_quad.grid)
            core_m = fit(inv_factors, F)
            gs = GS.from_sinc_1_over_sqrtx(46, 1.9)
            
            tg = TensorGrid(*([jnp.linspace(-0.5, 0.5, n)] * 3))
            elm_sp = BSpline(tg, degree=degree)
            tg_quad = TensorGrid(*([jnp.linspace(-0.5, 0.5, 2)] * 3)).to_gauss(self.quadrature_points)
            #tg_quad = tg.to_gauss(5)
            #inv_factors = elm_sp.pinv(tg_quad, alpha=1e-12)
            inv_factors = elm_sp.pinv(tg_quad, tol=1e-12)

            _, partials = elm_sp.factors_and_partials(tg_quad)
            sp_factors, info = superpotential_factors(elm_m, tg_quad, tg_m, gs, 
                                                      epsabs=1e-14, epsrel=0.0, order=31, max_ninter=200)
            _core = fit_superpotential(inv_factors, sp_factors, core_m, gs)
            _core = fit_divergence(inv_factors, partials, _core)
            _core = fit_laplace(inv_factors, partials, _core)
            core_h = fit_grad(inv_factors, partials, _core)
            
            factors_h = elm_sp.factors(tg_quad, mul_weights=True)
            factors_m = elm_m.factors(tg_quad)
            e = -1 / (16 * jnp.pi) * tucker_dot(
                TuckerTensor(core_h, factors_h),
                TuckerTensor(core_m, factors_m),
            )
            return e, info
        
        e, info = _energy()
        
        print(info)
        print(e, jnp.abs(e - 1 / 6))
        
        assert False

    @pytest.mark.parametrize("n", [10, 20, 40, 80])
    @pytest.mark.parametrize("k", [4, 5, 6, 7])
    @pytest.mark.parametrize("device", ["gpu"])
    def test_energy_flower_state(self, n, k, device, artifact_dir):
        if device == "gpu" and not jax.devices("gpu"):
            pytest.skip("GPU not available")

        csv_file = artifact_dir / f"energy_flower_{device}.csv" 

        #@partial(jax.jit, device=jax.devices(device)[0])
        @jax.jit
        def setup(quad_grid: TensorGrid):
            grid = TensorGrid(*([jnp.linspace(-0.5, 0.5, n)] * 3))
            solver = StrayFieldSolver.create(
                BSpline(grid, degree=k-1),
                quad_grid=quad_grid,
                gk_max_ninter=n + 50
            )
            return solver, solver.fit_mag(flower_state).tt
        
        #@partial(jax.jit, device=jax.devices(device)[0])
        @jax.jit
        def solve(solver, mag):
            mag = solver.fit_mag(mag)
            return mag.energy()
        
        quad_grid = TensorGrid(*([jnp.linspace(-0.5, 0.5, 2)] * 3)).to_gauss(self.quadrature_points)
        solver, mag = setup(quad_grid)
        runs = 1
        setup_time = timeit.timeit(lambda: setup(quad_grid)[1].core.block_until_ready(), number=runs) / runs
        e = solve(solver, mag)
        runs = 10
        run_time = timeit.timeit(lambda: solve(solver, mag).block_until_ready(), number=runs) / runs

        data = {
            "k": k,
            "knots": n,
            "energy": e,
            "setup_time": setup_time,
            "run_time": run_time
        }
        write_csv_row(csv_file, data)

    @pytest.mark.parametrize("n", [10, 20, 40, 80])
    @pytest.mark.parametrize("k", [4, 5, 6, 7])
    @pytest.mark.parametrize("device", ["gpu"])
    def test_energy_vortex_state(self, n, k, device, artifact_dir):
        if device == "gpu" and not jax.devices("gpu"):
            pytest.skip("GPU not available")

        csv_file = artifact_dir / f"energy_vortex_{device}.csv" 

        #@partial(jax.jit, device=jax.devices(device)[0])
        @jax.jit
        def setup(quad_grid: TensorGrid):
            grid = TensorGrid(*([jnp.linspace(-0.5, 0.5, n)] * 3))
            solver = StrayFieldSolver.create(
                BSpline(grid, degree=k-1),
                quad_grid=quad_grid,
                gk_max_ninter=n + 50
            )
            return solver, solver.fit_mag(vortex_state).tt
        
        #@partial(jax.jit, device=jax.devices(device)[0])
        @jax.jit
        def solve(solver, mag):
            mag = solver.fit_mag(mag)
            return mag.energy()
        
        quad_grid = TensorGrid(*([jnp.linspace(-0.5, 0.5, 2)] * 3)).to_gauss(self.quadrature_points)
        solver, mag = setup(quad_grid)
        runs = 1
        setup_time = timeit.timeit(lambda: setup(quad_grid)[1].core.block_until_ready(), number=runs) / runs
        e = solve(solver, mag)
        runs = 10
        run_time = timeit.timeit(lambda: solve(solver, mag).block_until_ready(), number=runs) / runs

        data = {
            "k": k,
            "knots": n,
            "energy": e,
            "setup_time": setup_time,
            "run_time": run_time
        }
        write_csv_row(csv_file, data)
