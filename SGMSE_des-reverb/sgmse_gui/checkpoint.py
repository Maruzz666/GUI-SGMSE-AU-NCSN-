import json
import subprocess
from pathlib import Path
from typing import Any

from .runtime import (
    launch_command,
    project_root,
    subprocess_environment,
    worker_creationflags,
)
from .backends import available_directml_devices


INFO_PREFIX = "SGMSE_GUI_MODEL_INFO="


def inspect_checkpoint(checkpoint: Path) -> dict[str, Any]:
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"No existe el checkpoint: {checkpoint}")

    process = subprocess.run(
        launch_command(["--inspect-checkpoint", str(checkpoint)]),
        cwd=project_root(),
        env=subprocess_environment(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        creationflags=worker_creationflags(),
    )
    if process.returncode:
        detail = "\n".join(
            line for line in (process.stdout + "\n" + process.stderr).splitlines()
            if line.strip()
        )
        raise RuntimeError(
            f"No se pudo cargar el checkpoint con ScoreModel.load_from_checkpoint "
            f"(código {process.returncode}).\n{detail[-6000:]}"
        )

    for line in reversed(process.stdout.splitlines()):
        if line.startswith(INFO_PREFIX):
            information = json.loads(line[len(INFO_PREFIX):])
            information["checkpoint_path"] = str(checkpoint)
            return information
    raise RuntimeError(
        "El proceso terminó sin devolver los metadatos del modelo.\n"
        f"{(process.stdout + process.stderr)[-6000:]}"
    )


def emit_checkpoint_information(checkpoint: Path) -> None:
    import torch

    from sgmse.model import ScoreModel

    checkpoint = checkpoint.expanduser().resolve()
    model = ScoreModel.load_from_checkpoint(str(checkpoint), map_location="cpu")
    model.eval()
    data = model.data_module
    backbone = str(model.backbone)
    sde_name = model.sde.__class__.__name__
    cuda_devices = []
    if torch.cuda.is_available():
        cuda_devices = [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]

    information = {
        "checkpoint_name": checkpoint.name,
        "backbone": backbone,
        "sde": sde_name,
        "score_model_sr": int(model.sr),
        "pipeline_sample_rate": 48000 if backbone == "ncsnpp_48k" else 16000,
        "n_fft": int(data.n_fft),
        "hop_length": int(data.hop_length),
        "transform_type": str(data.transform_type),
        "spec_factor": float(data.spec_factor),
        "spec_abs_exponent": float(data.spec_abs_exponent),
        "time_epsilon": float(model.t_eps),
        "sde_steps": int(model.sde.N),
        "cuda_devices": cuda_devices,
        "directml_devices": available_directml_devices(),
        "supports_inference": sde_name in {"OUVESDE", "SBVESDE"},
    }
    print(INFO_PREFIX + json.dumps(information, ensure_ascii=False))
