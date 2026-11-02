import runpy
import sys
from pathlib import Path

from .checkpoint import emit_checkpoint_information
from .runtime import project_root


def main() -> int:
    arguments = sys.argv[1:]
    if arguments and arguments[0] == "--inspect-checkpoint":
        if len(arguments) != 2:
            raise SystemExit("--inspect-checkpoint requiere la ruta del checkpoint.")
        emit_checkpoint_information(Path(arguments[1]))
        return 0
    if arguments and arguments[0] == "--sgmse-worker":
        script = project_root() / "enhancement.py"
        if not script.is_file():
            raise FileNotFoundError(f"No se encontró el pipeline original: {script}")
        sys.argv = [str(script), *arguments[1:]]
        runpy.run_path(str(script), run_name="__main__")
        return 0

    from .app import run_app

    run_app()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
