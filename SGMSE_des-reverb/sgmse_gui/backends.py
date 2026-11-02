from __future__ import annotations

from typing import Any


CPU_DEVICE = "cpu"
CUDA_DEVICE = "cuda"
DIRECTML_DEVICE = "directml"
HYBRID_DEVICE = "hybrid"
DIRECTML_BACKENDS = frozenset({DIRECTML_DEVICE, HYBRID_DEVICE})


def available_directml_devices() -> list[str]:
    try:
        import torch_directml
    except ModuleNotFoundError:
        return []

    return [
        torch_directml.device_name(index).rstrip("\x00 ")
        for index in range(torch_directml.device_count())
    ]


def configure_model_device(model: Any, requested_device: str) -> str:
    import torch

    if requested_device not in DIRECTML_BACKENDS:
        return requested_device

    if model.backbone != "ncsnpp_48k":
        raise ValueError(
            "La aceleración DirectML está validada únicamente para el backbone "
            "ncsnpp_48k; use CPU para este checkpoint."
        )
    if model.sde.__class__.__name__ != "OUVESDE":
        raise ValueError(
            "La aceleración DirectML requiere OUVESDE; los demás SDE aún "
            "no tienen una ruta de inferencia validada."
        )

    devices = available_directml_devices()
    if not devices:
        raise RuntimeError(
            "No se encontró una GPU compatible con DirectML en este entorno."
        )

    import torch_directml
    from sgmse.backbones.ncsnpp_utils.op.upfirdn2d import (
        set_hybrid_cpu_fallback,
    )

    compute_device = torch_directml.device(0)
    # eval() has applied EMA weights; keep its shadow copies and the SDE on CPU.
    model.dnn.to(compute_device)
    model.dnn.compute_device = compute_device
    set_hybrid_cpu_fallback(requested_device == HYBRID_DEVICE)
    return str(torch.device("cpu"))
