# tests/functional/conftest.py
import gc

import jax
import pytest
from datetime import datetime
from pathlib import Path


def pytest_addoption(parser):
    parser.addoption(
        "--device",
        action="store",
        default="gpu",
        #choices=["cpu", "gpu"],
        help="Select device to run JAX tests on.",
    )


@pytest.fixture(scope="session")
def device(request):
    device = request.config.getoption("--device")
    try:
        return jax.devices(device)[0]
    except RuntimeError:
        return jax.devices("cpu")[0]


@pytest.fixture(scope="session")
def artifact_dir(request):
    """Create an artifact directory for this pytest run (only if functional tests are selected)."""
    # Check if any functional tests are being run
    has_functional = any(item.get_closest_marker("functional") for item in request.session.items)
    if not has_functional:
        # No functional tests in this run
        yield None
        return

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = Path("artifacts") / "functional" / timestamp
    path.mkdir(parents=True, exist_ok=True)
    print(f"\n[functional setup] Created artifact dir: {path}")

    yield path  # pass to tests

    print(f"[functional teardown] Finished functional tests in: {path}")


@pytest.fixture(autouse=True)
def clean_jax():
    yield
    gc.collect()
    jax.clear_caches()
    jax.random.uniform(jax.random.key(0), ()).block_until_ready()