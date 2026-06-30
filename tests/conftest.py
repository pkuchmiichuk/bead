"""Root pytest configuration for bead package tests.

Adds a ``--run-integration`` flag and an ``integration`` marker. Integration
tests (which stand up a real PDS in docker) are deselected by default; passing
``--run-integration`` opts in. The ``pds_server`` fixture starts the PDS defined
in ``tests/pds`` and skips cleanly when docker, the image, or the health check
are unavailable.
"""

from __future__ import annotations

import os
import secrets
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

import didactic.api as dx
import httpx
import pytest

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "slow_model_training: marks tests that train ML models "
        "(deselect with '-m \"not slow_model_training\"')",
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests that need real IO such as a docker PDS "
        "(deselected unless --run-integration is given)",
    )


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the ``--run-integration`` command-line flag."""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="run integration tests (real IO, docker, credentials)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: Iterable[pytest.Item],
) -> None:
    """Skip integration tests unless ``--run-integration`` was passed."""
    if config.getoption("--run-integration"):
        return
    skip = pytest.mark.skip(reason="need --run-integration to run")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def tests_dir() -> Path:
    """Get tests directory path.

    Returns
    -------
    Path
        Path to tests directory
    """
    return Path(__file__).parent


@pytest.fixture
def sample_data_dir(tmp_path: Path) -> Path:
    """Create temporary directory for test data.

    Parameters
    ----------
    tmp_path : Path
        Pytest's tmp_path fixture

    Returns
    -------
    Path
        Path to temporary data directory
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


# --- PDS integration harness -------------------------------------------------

_PDS_COMPOSE = Path(__file__).parent / "pds" / "docker-compose.yml"
_PDS_HEALTH_TIMEOUT_S = 90.0


class PdsServer(dx.Model):
    """Connection details for a running local PDS test instance.

    Attributes
    ----------
    endpoint : str
        The base URL of the running PDS.
    did : str
        The DID of the throwaway account created for the test session.
    handle : str
        The account handle.
    password : str
        The account password.
    access_jwt : str
        The access token authorising writes to the account's repository.
    admin_password : str
        The PDS admin password generated for this session.
    """

    endpoint: str
    did: str
    handle: str
    password: str
    access_jwt: str
    admin_password: str


def _pds_port() -> int:
    override = os.environ.get("BEAD_PDS_PORT")
    if override:
        return int(override)
    with socket.socket() as probe:
        probe.bind(("", 0))
        return int(probe.getsockname()[1])


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    probe = subprocess.run(["docker", "info"], capture_output=True, check=False)
    return probe.returncode == 0


def _compose(
    args: Iterable[str], env: dict[str, str]
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["docker", "compose", "-f", str(_PDS_COMPOSE), *args],
        capture_output=True,
        check=False,
        env=env,
    )


def _wait_healthy(endpoint: str, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            response = httpx.get(f"{endpoint}/xrpc/_health", timeout=5.0)
        except httpx.HTTPError:
            response = None
        ok = response is not None and response.status_code == httpx.codes.OK
        if ok:
            return True
        time.sleep(2.0)
    return False


def create_pds_account(endpoint: str, admin_password: str) -> PdsServer:
    """Create a throwaway account on the PDS and return its connection details."""
    token = secrets.token_hex(8)
    handle = f"u{token}.test"
    password = secrets.token_hex(16)
    response = httpx.post(
        f"{endpoint}/xrpc/com.atproto.server.createAccount",
        json={
            "handle": handle,
            "email": f"{token}@example.test",
            "password": password,
        },
        timeout=30.0,
    )
    response.raise_for_status()
    body = response.json()
    return PdsServer(
        endpoint=endpoint,
        did=str(body["did"]),
        handle=str(body.get("handle", handle)),
        password=password,
        access_jwt=str(body["accessJwt"]),
        admin_password=admin_password,
    )


@pytest.fixture(scope="session")
def pds_server() -> Iterator[PdsServer]:
    """Yield a running local PDS with a throwaway account, or skip cleanly."""
    if not _docker_available():
        pytest.skip("docker is not available")
    port = _pds_port()
    endpoint = f"http://localhost:{port}"
    admin_password = secrets.token_hex(16)
    env = {
        **os.environ,
        "BEAD_PDS_PORT": str(port),
        "PDS_JWT_SECRET": secrets.token_hex(16),
        "PDS_ADMIN_PASSWORD": admin_password,
        "PDS_PLC_ROTATION_KEY_K256_PRIVATE_KEY_HEX": secrets.token_hex(32),
    }
    started = _compose(["up", "-d"], env)
    if started.returncode != 0:
        pytest.skip(
            f"could not start the pds container: {started.stderr.decode()[:300]}"
        )
    try:
        if not _wait_healthy(endpoint, _PDS_HEALTH_TIMEOUT_S):
            pytest.skip("the pds container did not become healthy in time")
        try:
            server = create_pds_account(endpoint, admin_password)
        except httpx.HTTPError as exc:
            pytest.skip(f"could not create a test account: {exc}")
        yield server
    finally:
        _compose(["down", "-v"], env)
