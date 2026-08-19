"""Optional MATLAB adapters for the unchanged Liéois reference code."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

from ..simulation.static_sem import EstimatorResult

UPSTREAM_URL = "https://github.com/CyclotronResearchCentre/SparseLowRankIdentification"
UPSTREAM_COMMIT = "f9ca591039fb5c6a3288108e371ba328c7284e48"


class LiegeoisSubprocessError(RuntimeError):
    """MATLAB batch failure with the process diagnostics attached."""

    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def matlab_engine_available():
    try:
        import matlab.engine  # noqa: F401
        return True
    except (ImportError, ModuleNotFoundError):
        return False


def matlab_subprocess_executable(matlab_executable=None):
    """Return the configured MATLAB executable, or ``None`` when unavailable."""
    if matlab_executable is not None:
        candidate = Path(matlab_executable).expanduser()
        return str(candidate.resolve()) if candidate.is_file() else None
    return shutil.which("matlab") or shutil.which("matlab.exe")


def matlab_subprocess_available(matlab_executable=None):
    return matlab_subprocess_executable(matlab_executable) is not None


def _repo_commit(repo):
    try:
        return subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _array(value):
    try:
        return np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return value


def _native_struct(value):
    if hasattr(value, "_fieldnames"):
        return {name: _native_struct(getattr(value, name)) for name in value._fieldnames}
    if isinstance(value, dict):
        return {str(k): _native_struct(v) for k, v in value.items()}
    return _array(value)


def _matlab_quote(path):
    return str(Path(path).resolve()).replace("'", "''")


def _matlab_text(value):
    value = np.asarray(value).squeeze()
    if value.dtype.kind in "US":
        return "".join(value.reshape(-1).tolist())
    return str(value.item() if value.ndim == 0 else value)


def _process_text(value):
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value or ""


def _validate_request(X, p, repo_path, validate_commit):
    X = np.asarray(X, float)
    if X.ndim != 2 or p < 1 or p >= len(X) or not np.isfinite(X).all():
        raise ValueError("invalid X or p")
    root = Path(__file__).resolve().parents[3]
    repo = Path(repo_path).resolve() if repo_path else root / "external" / "SparseLowRankIdentification"
    if not (repo / "ADMM_sparse_lowrank_AR.m").exists():
        raise RuntimeError(f"Liéois upstream repository unavailable at {repo}")
    commit = _repo_commit(repo)
    if validate_commit and commit != UPSTREAM_COMMIT:
        raise RuntimeError(
            f"Liéois upstream commit mismatch: expected {UPSTREAM_COMMIT}, found {commit}"
        )
    return X, root, repo, commit


def _result(native, backend, commit, matlab_version, runtime, process_diagnostics=None,
            lambda_value=.004, gamma=1.25):
    xs = native["x_sol"]
    for key in ("L", "S", "Omega", "Delta", "h"):
        if key not in xs:
            raise RuntimeError(f"Liéois output missing x_sol.{key}")
    h = int(np.asarray(xs["h"]).squeeze())
    shapes = {key: list(np.shape(xs[key])) for key in ("L", "S", "Omega", "Delta")}
    diagnostics = {
        "baseline": "Liéois original MATLAB implementation",
        "backend": backend,
        "upstream_url": UPSTREAM_URL,
        "upstream_commit": commit,
        "matlab_version": matlab_version,
        "lambda": lambda_value,
        "gamma": gamma,
        "estimated_latent_dimension": h,
        "native_shapes": shapes,
        "runtime_seconds": runtime,
    }
    if process_diagnostics:
        diagnostics.update(process_diagnostics)
    return EstimatorResult(
        graph=np.asarray(xs["Omega"]), native_result=native,
        runtime_seconds=runtime, diagnostics=diagnostics,
    )


def fit_liegeois(X, p=1, backend="matlab_subprocess", repo_path=None, engine=None,
                 matlab_executable=None, timeout=3600,
                 lambda_value=.004, gamma=1.25, maxiter=1000, rho_max=1e3,
                 abstol=1e-4, reltol=1e-4, compute_relative_duality_gap=True,
                 compute_primal_variables=True, verb=False, validate_commit=True):
    """Execute upstream ``ADMM_sparse_lowrank_AR`` via either MATLAB backend.

    The subprocess backend is the batch-oriented default. The Engine backend is
    retained for interactive use. Returned values are the native sparse/low-rank
    graphical-model objects, not directed ``W0``/``W_lags`` estimates.
    """
    if backend not in {"matlab_subprocess", "matlab_engine"}:
        raise ValueError("backend must be 'matlab_subprocess' or 'matlab_engine'")
    X, root, repo, commit = _validate_request(X, p, repo_path, validate_commit)

    if backend == "matlab_engine":
        if engine is None and not matlab_engine_available():
            raise ImportError("MATLAB Engine for Python is unavailable")
        import matlab
        owns_engine = engine is None
        if owns_engine:
            import matlab.engine
            engine = matlab.engine.start_matlab()
        bridge = root / "external_wrappers" / "liegeois"
        engine.addpath(str(bridge), nargout=0)
        started = time.perf_counter()
        try:
            x_sol, infos, used_options, C = engine.run_liegeois_engine(
                matlab.double(X.tolist()), float(p), str(repo), float(lambda_value), float(gamma),
                float(maxiter), float(rho_max), float(abstol), float(reltol),
                bool(compute_relative_duality_gap), bool(compute_primal_variables), bool(verb),
                nargout=4,
            )
            matlab_version = str(engine.version())
        finally:
            if owns_engine:
                engine.quit()
        runtime = time.perf_counter() - started
        native = {
            "x_sol": _native_struct(x_sol), "infos": _native_struct(infos),
            "used_options": _native_struct(used_options), "C": _array(C),
        }
        return _result(native, backend, commit, matlab_version, runtime,
                       lambda_value=lambda_value, gamma=gamma)

    executable = matlab_subprocess_executable(matlab_executable)
    if executable is None:
        detail = f" at {matlab_executable}" if matlab_executable else " on PATH"
        raise ImportError(f"MATLAB executable is unavailable{detail}")
    bridge = root / "external_wrappers" / "liegeois"
    temp_root = root / ".baseline_tmp"
    temp_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="liegeois-", dir=temp_root) as temp_dir:
        input_path = Path(temp_dir) / "input.mat"
        output_path = Path(temp_dir) / "output.mat"
        savemat(input_path, {
            "X": X, "p": float(p), "lambda_value": float(lambda_value),
            "gamma": float(gamma), "maxiter": float(maxiter), "rho_max": float(rho_max),
            "abstol": float(abstol), "reltol": float(reltol),
            "compute_relative_duality_gap": bool(compute_relative_duality_gap),
            "compute_primal_variables": bool(compute_primal_variables), "verb": bool(verb),
        })
        expression = (
            f"addpath('{_matlab_quote(bridge)}'); "
            f"run_liegeois_batch('{_matlab_quote(input_path)}',"
            f"'{_matlab_quote(output_path)}','{_matlab_quote(repo)}');"
        )
        command = [executable, "-batch", expression]
        started = time.perf_counter()
        try:
            process = subprocess.run(
                command, capture_output=True, text=True, timeout=timeout, check=False,
            )
        except subprocess.TimeoutExpired as exc:
            runtime = time.perf_counter() - started
            diagnostics = {
                "command": command, "stdout": _process_text(exc.stdout),
                "stderr": _process_text(exc.stderr),
                "return_code": None, "timed_out": True, "timeout_seconds": timeout,
                "runtime_seconds": runtime,
            }
            raise LiegeoisSubprocessError(
                f"MATLAB subprocess timed out after {runtime:.1f} seconds", diagnostics
            ) from exc
        runtime = time.perf_counter() - started
        process_diagnostics = {
            "command": command, "stdout": process.stdout, "stderr": process.stderr,
            "return_code": process.returncode, "timed_out": False,
            "timeout_seconds": timeout, "matlab_executable": executable,
        }
        if process.returncode != 0:
            raise LiegeoisSubprocessError(
                f"MATLAB subprocess failed with return code {process.returncode}",
                {**process_diagnostics, "runtime_seconds": runtime},
            )
        if not output_path.exists():
            raise LiegeoisSubprocessError(
                "MATLAB subprocess did not produce its output MAT file",
                {**process_diagnostics, "runtime_seconds": runtime},
            )
        payload = loadmat(output_path, squeeze_me=True, struct_as_record=False)
        native = {
            "x_sol": _native_struct(payload["x_sol"]),
            "infos": _native_struct(payload["infos"]),
            "used_options": _native_struct(payload["used_options"]),
            "C": _array(payload["C"]),
        }
        matlab_version = _matlab_text(payload["matlab_version"])
    return _result(native, backend, commit, matlab_version, runtime, process_diagnostics,
                   lambda_value=lambda_value, gamma=gamma)


liegeois = fit_liegeois
