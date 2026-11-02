from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any


OPERATIONS: dict[str, Callable[[], Any]] = {}


def _register_operations() -> None:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch_directml
    from sgmse.backbones.ncsnpp_utils.op.upfirdn2d import upfirdn2d_directml

    device = torch_directml.device(0)
    x = torch.randn(1, 4, 16, 16, device=device)
    real = torch.randn(1, 1, 16, 16, device=device)
    waveform = torch.randn(512, device=device)
    fir_kernel = torch.ones(4, 4, device=device) / 16

    OPERATIONS.update(
        {
            "add": lambda: x + x,
            "abs": lambda: torch.abs(x),
            "sum": lambda: x.sum(),
            "vector_norm": lambda: torch.linalg.vector_norm(x),
            "log": lambda: torch.log(torch.abs(x) + 1),
            "exp": lambda: torch.exp(x),
            "sin": lambda: torch.sin(x),
            "sqrt": lambda: torch.sqrt(torch.abs(x) + 1),
            "randn_like": lambda: torch.randn_like(x),
            "linear": lambda: nn.Linear(16, 32).to(device)(torch.randn(1, 16, device=device)),
            "conv2d": lambda: nn.Conv2d(4, 8, 3, padding=1).to(device)(x),
            "conv_transpose2d": lambda: nn.ConvTranspose2d(4, 8, 3, padding=1).to(device)(x),
            "group_norm": lambda: nn.GroupNorm(4, 4).to(device)(x),
            "instance_norm": lambda: nn.InstanceNorm2d(4).to(device)(x),
            "relu": lambda: F.relu(x),
            "silu": lambda: F.silu(x),
            "leaky_relu": lambda: F.leaky_relu(x, 0.2),
            "gelu": lambda: F.gelu(x),
            "interpolate_nearest": lambda: F.interpolate(x, scale_factor=2, mode="nearest"),
            "interpolate_bilinear": lambda: F.interpolate(
                x, scale_factor=2, mode="bilinear", align_corners=True
            ),
            "reflection_pad": lambda: F.pad(x, (2, 2, 2, 2), mode="reflect"),
            "flip": lambda: torch.flip(fir_kernel, [0, 1]),
            "cat": lambda: torch.cat((x.unsqueeze(-1), torch.zeros_like(x).unsqueeze(-1)), dim=-1),
            "grid_sample": lambda: F.grid_sample(
                x, torch.zeros(1, 16, 16, 2, device=device)
            ),
            "upfirdn2d": lambda: upfirdn2d_directml(
                x, fir_kernel, 1, 1, 2, 2, 1, 1, 1, 1
            ),
            "view_as_complex": lambda: torch.view_as_complex(
                torch.stack((real, real), dim=-1).contiguous()
            ),
            "fft_rfft": lambda: torch.fft.rfft(real, dim=-1),
            "stft": lambda: torch.stft(
                waveform, n_fft=16, hop_length=4, return_complex=True
            ),
        }
    )


def _run_one(name: str) -> dict[str, Any]:
    import torch
    import torch_directml

    _register_operations()
    value = OPERATIONS[name]()
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Probe {name} returned {type(value).__name__}, not a tensor.")
    if value.is_complex():
        return {
            "operation": name,
            "ok": True,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    finite = bool(torch.isfinite(value).all().to("cpu").item())
    return {
        "operation": name,
        "ok": finite,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "finite": finite,
        "device_name": torch_directml.device_name(0).strip(),
    }


def _run_parent() -> int:
    import torch_directml

    count = torch_directml.device_count()
    if count < 1:
        raise RuntimeError("DirectML no encontró adaptadores compatibles.")

    results: list[dict[str, Any]] = []
    for operation in OPERATIONS:
        child = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--operation", operation],
            capture_output=True,
            text=True,
            check=False,
        )
        try:
            result = json.loads(child.stdout.splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            result = {
                "operation": operation,
                "ok": False,
                "return_code": child.returncode,
                "stdout": child.stdout[-2000:],
                "stderr": child.stderr[-2000:],
            }
        result["return_code"] = child.returncode
        results.append(result)

    print(
        json.dumps(
            {
                "devices": [
                    torch_directml.device_name(index).strip()
                    for index in range(count)
                ],
                "operations": results,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return int(any(not row.get("ok", False) for row in results))


def main() -> int:
    _register_operations()
    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", choices=tuple(OPERATIONS))
    args = parser.parse_args()
    if args.operation:
        try:
            result = _run_one(args.operation)
        except Exception as error:
            result = {
                "operation": args.operation,
                "ok": False,
                "error": f"{type(error).__name__}: {error}",
            }
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 0 if result["ok"] else 1

    return _run_parent()


if __name__ == "__main__":
    raise SystemExit(main())
