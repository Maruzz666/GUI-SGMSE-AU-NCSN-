import os
import subprocess
import sys
from pathlib import Path


def project_root() -> Path:
    if getattr(sys, "frozen", False):
        bundled_root = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
        if (bundled_root / "enhancement.py").is_file():
            return bundled_root
    return Path(__file__).resolve().parents[1]


def runtime_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return project_root()


def temp_root() -> Path:
    return runtime_root() / ".tmp"


def prepare_runtime() -> Path:
    root = temp_root()
    root.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    return root


def launch_command(arguments: list[str]) -> list[str]:
    if getattr(sys, "frozen", False):
        worker = Path(sys.executable).with_name("SGMSE-worker.exe")
        if not worker.is_file():
            raise FileNotFoundError(
                f"No se encontró el ejecutable auxiliar de SGMSE: {worker}"
            )
        return [str(worker), *arguments]
    return [sys.executable, "-m", "sgmse_gui", *arguments]


def worker_creationflags() -> int:
    if getattr(sys, "frozen", False) and os.name == "nt":
        return subprocess.CREATE_NO_WINDOW
    return 0


def subprocess_environment() -> dict[str, str]:
    root = project_root()
    temp = prepare_runtime()
    numba_cache = temp / "n"
    numba_cache.mkdir(exist_ok=True)
    environment = os.environ.copy()
    existing_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(root), existing_path] if existing_path else [str(root)]
    )
    environment["TEMP"] = str(temp)
    environment["TMP"] = str(temp)
    environment["NUMBA_CACHE_DIR"] = str(numba_cache)
    return environment
