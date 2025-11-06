import timeit
from typing import Callable

import pytest

from tpelm.bspline import BSpline
from tpelm.tensor_grid import TensorGrid
from tpelm.magnetostatic import DomainState, solve_energy, fit_mag

from .. import *
from ..sources import flower_state, vortex_state, m_uniform
from .utils import write_csv_row


def compute_mag(tg: TensorGrid, m: Callable) -> jax.Array:
    grid = tg.grid
    M = jax.lax.map(jax.vmap(jax.vmap(m)), grid)  # map along last dim to save memory
    return M


@pytest.mark.functional
class TestEnergy:
    quadrature_points = 140

    def setup_class(self):
        jax.config.update("jax_compilation_cache_dir", "/tmp/jax_test_cache")
        jax.config.update("jax_enable_x64", True)

    def teardown_class(self):
        jax.config.update("jax_enable_x64", False)

    @pytest.mark.parametrize("n", [10, 20, 40])
    @pytest.mark.parametrize("k", [3, 4, 5, 6, 7])
    def test_energy_uniform_cube(self, n, k, device, artifact_dir):
        with jax.default_device(device):
            s = 46

            csv_file = artifact_dir / f"energy_uniform.csv"
            grid = TensorGrid(*([jnp.linspace(-0.5, 0.5, n)] * 3))
            
            @jax.jit
            def setup(quad_grid: TensorGrid):
                mag_tg = TensorGrid(*([jnp.linspace(-0.5, 0.5, 2)] * 3))
                state = DomainState.init(
                    sp_elm=BSpline(grid, degree=k - 1),
                    mag_elm=BSpline(mag_tg, degree=0),
                    sp_quad_grid=quad_grid,
                    mag_quad_grid=mag_tg.to_gauss(1),
                    gs_terms=s,
                    gk_max_ninter=100
                )
                M = compute_mag(mag_tg.to_gauss(1), m_uniform)
                mag = fit_mag(state, M)
                return state, mag
            
            @jax.jit
            def solve(state: DomainState, mag) -> jax.Array:
                return solve_energy(state, mag, {0: quad_grid})
            
            quad_grid = grid.to_gauss(k)
            solver, mag = setup(quad_grid)
            runs = 1
            setup_time = timeit.timeit(lambda: setup(quad_grid)[1][0].core.block_until_ready(), number=runs) / runs
            e = solve(solver, mag)
            runs = 10
            run_time = timeit.timeit(lambda: solve(solver, mag).block_until_ready(), number=runs) / runs

            data = {
                "k": k,
                "r": n,
                "s": s,
                "energy": e,
                "error": jnp.abs(e - 1 / 6),
                "setup_time": setup_time,
                "run_time": run_time
            }
            write_csv_row(csv_file, data)
        
    @pytest.mark.parametrize("n", [10, 20, 40, 80])
    @pytest.mark.parametrize("k", [3, 4, 5, 6, 7])
    def test_energy_flower_state(self, n, k, device, artifact_dir):
        with jax.default_device(device):
            s = 46

            csv_file = artifact_dir / f"energy_flower.csv" 
            grid = TensorGrid(*([jnp.linspace(-0.5, 0.5, n)] * 3))
            
            @jax.jit
            def setup(quad_grid: TensorGrid):
                state = DomainState.init(
                    sp_elm=BSpline(grid, degree=k - 1),
                    sp_quad_grid=quad_grid,
                    gs_terms=s,
                    gk_max_ninter=n + 50
                )
                M = compute_mag(state.quad_grids[0], flower_state)
                mag = fit_mag(state, M)
                return state, mag
            
            @jax.jit
            def solve(state: DomainState, mag) -> jax.Array:
                return solve_energy(state, mag, {0: quad_grid})
            
            quad_grid = grid.to_gauss(k)
            solver, mag = setup(quad_grid)
            runs = 1
            setup_time = timeit.timeit(lambda: setup(quad_grid)[1][0].core.block_until_ready(), number=runs) / runs
            e = solve(solver, mag)
            runs = 10
            run_time = timeit.timeit(lambda: solve(solver, mag).block_until_ready(), number=runs) / runs

            data = {
                "k": k,
                "r": n,
                "s": s,
                "energy": e,
                "setup_time": setup_time,
                "run_time": run_time
            }
            write_csv_row(csv_file, data)

    @pytest.mark.parametrize("s", [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70])
    def test_energy_flower_state_s(self, s, device, artifact_dir):
        with jax.default_device(device):
            csv_file = artifact_dir / "energy_flower_s.csv"
            n = 50
            k = 10
            grid = TensorGrid(*([jnp.linspace(-0.5, 0.5, n)] * 3))
            
            @jax.jit
            def setup(quad_grid: TensorGrid):
                state = DomainState.init(
                    sp_elm=BSpline(grid, degree=k - 1),
                    sp_quad_grid=quad_grid,
                    gs_terms=s,
                    gk_max_ninter=150
                )
                M = compute_mag(state.quad_grids[0], flower_state)
                mag = fit_mag(state, M)
                return state, mag
            
            @jax.jit
            def solve(state: DomainState, mag) -> jax.Array:
                return solve_energy(state, mag, {0: quad_grid})
            
            quad_grid = grid.to_gauss(k)
            solver, mag = setup(quad_grid)
            runs = 1
            setup_time = timeit.timeit(lambda: setup(quad_grid)[1][0].core.block_until_ready(), number=runs) / runs
            e = solve(solver, mag)
            runs = 10
            run_time = timeit.timeit(lambda: solve(solver, mag).block_until_ready(), number=runs) / runs

            data = {
                "k": k,
                "r": n,
                "s": s,
                "energy": e,
                "setup_time": setup_time,
                "run_time": run_time
            }
            write_csv_row(csv_file, data)
        
    @pytest.mark.parametrize("n", [10, 20, 40, 80])
    @pytest.mark.parametrize("k", [3, 4, 5, 6, 7])
    def test_energy_vortex_state(self, n, k, device, artifact_dir):
        with jax.default_device(device):
            s = 46

            csv_file = artifact_dir / f"energy_vortex.csv" 
            grid = TensorGrid(*([jnp.linspace(-0.5, 0.5, n)] * 3))
            
            @jax.jit
            def setup(quad_grid: TensorGrid):
                state = DomainState.init(
                    sp_elm=BSpline(grid, degree=k - 1),
                    sp_quad_grid=quad_grid,
                    gs_terms=s,
                    gk_max_ninter=n + 50
                )
                M = compute_mag(state.quad_grids[0], vortex_state)
                mag = fit_mag(state, M)
                return state, mag
            
            @jax.jit
            def solve(state: DomainState, mag) -> jax.Array:
                return solve_energy(state, mag, {0: quad_grid})
            
            quad_grid = grid.to_gauss(k)
            solver, mag = setup(quad_grid)
            runs = 1
            setup_time = timeit.timeit(lambda: setup(quad_grid)[1][0].core.block_until_ready(), number=runs) / runs
            e = solve(solver, mag)
            runs = 10
            run_time = timeit.timeit(lambda: solve(solver, mag).block_until_ready(), number=runs) / runs

            data = {
                "k": k,
                "r": n,
                "s": s,
                "energy": e,
                "setup_time": setup_time,
                "run_time": run_time
            }
            write_csv_row(csv_file, data)

    @pytest.mark.parametrize("s", [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70])
    def test_energy_vortex_state_s(self, s, device, artifact_dir):
        with jax.default_device(device):
            csv_file = artifact_dir / "energy_vortex_s.csv"
            n = 50
            k = 10
            grid = TensorGrid(*([jnp.linspace(-0.5, 0.5, n)] * 3))
            
            @jax.jit
            def setup(quad_grid: TensorGrid):
                state = DomainState.init(
                    sp_elm=BSpline(grid, degree=k - 1),
                    sp_quad_grid=quad_grid,
                    gs_terms=s,
                    gk_max_ninter=150
                )
                M = compute_mag(state.quad_grids[0], vortex_state)
                mag = fit_mag(state, M)
                return state, mag
            
            @jax.jit
            def solve(state: DomainState, mag) -> jax.Array:
                return solve_energy(state, mag, {0: quad_grid})
            
            quad_grid = grid.to_gauss(k)
            solver, mag = setup(quad_grid)
            runs = 1
            setup_time = timeit.timeit(lambda: setup(quad_grid)[1][0].core.block_until_ready(), number=runs) / runs
            e = solve(solver, mag)
            runs = 10
            run_time = timeit.timeit(lambda: solve(solver, mag).block_until_ready(), number=runs) / runs

            data = {
                "S": s,
                "k": k,
                "knots": n,
                "energy": e,
                "setup_time": setup_time,
                "run_time": run_time
            }
            write_csv_row(csv_file, data)

    @pytest.mark.parametrize("n", [20, 40, 60])
    @pytest.mark.parametrize("k", [4, 5, 6, 7, 8])
    def test_energy_vortex_thin_film(self, n, k, device, artifact_dir):
        with jax.default_device(device):
            s = 46
            csv_file = artifact_dir / f"energy_vortex_thin_film.csv" 
            grid = TensorGrid(
                jnp.linspace(-0.5, 0.5, n),
                jnp.linspace(-0.5, 0.5, n),
                jnp.linspace(-0.05, 0.05, n // 4)
            )
            
            @jax.jit
            def setup(quad_grid: TensorGrid):
                state = DomainState.init(
                    sp_elm=BSpline(grid, degree=k - 1),
                    sp_quad_grid=quad_grid,
                    gs_terms=s,
                    gk_max_ninter=n + 50
                )
                mag = fit_mag(state, vortex_state)
                return state, mag
            
            @jax.jit
            def solve(state: DomainState, mag) -> jax.Array:
                return solve_energy(state, mag, {0: quad_grid})
            
            quad_grid = grid.to_gauss(k)
            solver, mag = setup(quad_grid)
            runs = 1
            setup_time = timeit.timeit(lambda: setup(quad_grid)[1][0].core.block_until_ready(), number=runs) / runs
            e = solve(solver, mag)
            runs = 10
            run_time = timeit.timeit(lambda: solve(solver, mag).block_until_ready(), number=runs) / runs

            data = {
                "k": k,
                "r1": n,
                "r2": n,
                "r3": n // 4,
                "s": s,
                "energy": e,
                "setup_time": setup_time,
                "run_time": run_time
            }
            write_csv_row(csv_file, data)

    @pytest.mark.parametrize("n", [20, 40, 60])
    @pytest.mark.parametrize("k", [4, 5, 6, 7, 8])
    def test_energy_two_layer(self, n, k, device, artifact_dir):
        r"""This test computes the energy for two stacked thin films
        :math:`A=[-0.4, 0.5]\times[-0.5, 0.5]\times[0, 0.05]` and 
        :math:`B=[-0.5, 0.4]\times[-0.5, 0.5]\times[-0.05, 0]`
        with the layer A having vortex magnetization with Ms=1 and B having
        uniform magnetization in x with Ms=2.
        """
        with jax.default_device(device):
            s = 46
            csv_file = artifact_dir / f"energy_two_layer.csv" 
            
            grid_a = TensorGrid(
                jnp.linspace(-0.4, 0.5, n),
                jnp.linspace(-0.5, 0.5, n),
                jnp.linspace(0.0, 0.05, n // 4)
            )
            grid_b = TensorGrid(
                jnp.linspace(-0.5, 0.4, n),
                jnp.linspace(-0.5, 0.5, n),
                jnp.linspace(-0.05, 0.0, n // 4)
            )
            
            @jax.jit
            def setup(quad_grid: dict[int, TensorGrid]):
                mag_grid_a = TensorGrid(
                    jnp.linspace(-0.4, 0.5, 2),
                    jnp.linspace(-0.5, 0.5, 2),
                    jnp.linspace(0.0, 0.05, 2)
                )
                mag_grid_b = TensorGrid(
                    jnp.linspace(-0.5, 0.4, 2),
                    jnp.linspace(-0.5, 0.5, 2),
                    jnp.linspace(-0.05, 0.0, 2)
                )
                
                state = DomainState.init(
                    sp_elm={
                        0: BSpline(grid_a, degree=k - 1),
                        1: BSpline(grid_b, degree=k - 1),
                    },
                    mag_elm={
                        0: BSpline(mag_grid_a, degree=0),
                        1: BSpline(mag_grid_b, degree=0),
                    },
                    mag_quad_grid={0: mag_grid_a.to_gauss(1), 1: mag_grid_b.to_gauss(1)},
                    sp_quad_grid=quad_grid,
                    gs_terms=s,
                    gk_max_ninter=50
                )
                _m1 = jnp.array([0.2, 0.5, 0.3])
                _m2 = jnp.array([-1, -0.3, 0.0])
                _m1 = _m1 / jnp.linalg.norm(_m1)
                _m2 = _m2 / jnp.linalg.norm(_m2) * 2
                m1 = lambda x: jnp.zeros_like(x).at[..., :].set(_m1)
                m2 = lambda x: jnp.zeros_like(x).at[..., :].set(_m2)
                _mags = {
                    0: m1,
                    1: m2
                }
                mag = fit_mag(state, _mags)
                return state, mag
            
            @jax.jit
            def solve(state: DomainState, mag) -> jax.Array:
                return solve_energy(state, mag, quad_grid)
            
            quad_grid = {
                0: grid_a.to_gauss(k),
                1: grid_b.to_gauss(k)
            }
            solver, mag = setup(quad_grid)
            runs = 1
            setup_time = timeit.timeit(lambda: setup(quad_grid)[1][0].core.block_until_ready(), number=runs) / runs
            e = solve(solver, mag)
            runs = 10
            run_time = timeit.timeit(lambda: solve(solver, mag).block_until_ready(), number=runs) / runs

            data = {
                "k": k,
                "r1": n,
                "r2": n,
                "r3": n // 2,
                "s": s,
                "energy": e,
                "setup_time": setup_time,
                "run_time": run_time
            }
            write_csv_row(csv_file, data)
