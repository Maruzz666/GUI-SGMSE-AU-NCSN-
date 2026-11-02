import json
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .audio import AudioItem, AudioMetadata, inspect_audio, validate_output_names
from .history import append_history
from .runtime import (
    launch_command,
    prepare_runtime,
    project_root,
    subprocess_environment,
    worker_creationflags,
)
from .backends import DIRECTML_BACKENDS


@dataclass(frozen=True)
class InferenceSettings:
    sampler_type: str = "pc"
    corrector: str = "ald"
    corrector_steps: int = 1
    snr: float = 0.5
    steps: int = 30
    device: str = "cpu"
    time_epsilon: float = 0.03
    seed: int | None = None


@dataclass(frozen=True)
class TaskEvent:
    kind: str
    message: str = ""
    completed: int = 0
    total: int = 0
    payload: Any = None


@dataclass(frozen=True)
class ProcessedAudio:
    source: Path
    output: Path
    metadata: AudioMetadata


class TaskCancelled(Exception):
    pass


def validate_settings(settings: InferenceSettings) -> None:
    if settings.device != "cpu" and not settings.device.startswith("cuda"):
        if settings.device not in DIRECTML_BACKENDS:
            raise ValueError(f"Dispositivo de inferencia desconocido: {settings.device}.")
    if settings.sampler_type not in {"pc", "ode"}:
        raise ValueError("El sampler debe ser PC u ODE.")
    if settings.corrector not in {"ald", "langevin", "none"}:
        raise ValueError("El corrector debe ser ALD, Langevin o ninguno.")
    if settings.steps < 1:
        raise ValueError("N debe ser al menos 1.")
    if settings.corrector_steps < 0:
        raise ValueError("Los pasos del corrector no pueden ser negativos.")
    if settings.snr <= 0:
        raise ValueError("SNR debe ser mayor que cero.")
    if not 0 < settings.time_epsilon < 1:
        raise ValueError("t_eps debe estar entre 0 y 1.")


def build_enhancement_arguments(
    input_directory: Path,
    output_directory: Path,
    checkpoint: Path,
    settings: InferenceSettings,
) -> list[str]:
    arguments = [
        "--test_dir",
        str(input_directory),
        "--enhanced_dir",
        str(output_directory),
        "--ckpt",
        str(checkpoint),
        "--sampler_type",
        settings.sampler_type,
        "--corrector",
        settings.corrector,
        "--corrector_steps",
        str(settings.corrector_steps),
        "--snr",
        str(settings.snr),
        "--N",
        str(settings.steps),
        "--device",
        settings.device,
        "--t_eps",
        str(settings.time_epsilon),
    ]
    if settings.seed is not None:
        arguments.extend(["--seed", str(settings.seed)])
    return arguments


def validate_job(
    items: list[AudioItem],
    checkpoint: Path,
    output_directory: Path,
    settings: InferenceSettings,
    model_info: dict[str, Any],
) -> list[tuple[AudioItem, AudioMetadata, Path]]:
    validate_settings(settings)
    validate_output_names(items)
    if not items:
        raise ValueError("Selecciona al menos un archivo de audio.")
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"No existe el checkpoint: {checkpoint}")
    if Path(str(model_info.get("checkpoint_path", ""))).resolve() != checkpoint:
        raise ValueError("Inspecciona el checkpoint seleccionado antes de procesar.")
    if not model_info.get("supports_inference", False):
        raise ValueError(
            f"El SDE {model_info.get('sde', 'desconocido')} no está soportado "
            "por enhancement.py."
        )

    sde_name = model_info["sde"]
    if sde_name not in {"OUVESDE", "SBVESDE"}:
        raise ValueError(f"SDE no compatible con enhancement.py: {sde_name}")
    if sde_name == "SBVESDE" and not settings.device.startswith("cuda"):
        raise ValueError(
            "El camino SBVESDE de enhancement.py usa Y.cuda() directamente; "
            "selecciona CUDA o un checkpoint OUVESDE."
        )
    if settings.device.startswith("cuda") and not model_info.get("cuda_devices"):
        raise ValueError("CUDA no está disponible en el entorno de inferencia.")
    if settings.device in DIRECTML_BACKENDS:
        if not model_info.get("directml_devices"):
            raise ValueError("DirectML no está disponible en el entorno de inferencia.")
        if model_info.get("backbone") != "ncsnpp_48k":
            raise ValueError(
                "La inferencia DirectML está validada solo para ncsnpp_48k."
            )
        if sde_name != "OUVESDE":
            raise ValueError(
                "La inferencia DirectML requiere un checkpoint OUVESDE."
            )

    expected_channels = 1
    output_directory = output_directory.expanduser().resolve()
    validated: list[tuple[AudioItem, AudioMetadata, Path]] = []
    for item in items:
        metadata = inspect_audio(item.source)
        if metadata.channels != expected_channels:
            raise ValueError(
                f"{item.source.name} tiene {metadata.channels} canales. "
                f"Este checkpoint/backbone espera {expected_channels}; no se "
                "mezclarán ni eliminarán canales."
            )
        destination = output_directory / item.relative_output
        if destination.exists():
            raise FileExistsError(
                f"El resultado ya existe y no se sobrescribirá: {destination}"
            )
        validated.append((item, metadata, destination))
    return validated


def _record(
    status: str,
    checkpoint: Path,
    output_directory: Path,
    settings: InferenceSettings,
    started_at: str,
    files: list[dict[str, str]],
    error: str | None = None,
) -> None:
    entry: dict[str, Any] = {
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "checkpoint": str(checkpoint),
        "output_directory": str(output_directory),
        "settings": asdict(settings),
        "files": files,
    }
    if error:
        entry["error"] = error
    append_history(entry)


def run_inference(
    items: list[AudioItem],
    checkpoint: Path,
    output_directory: Path,
    settings: InferenceSettings,
    model_info: dict[str, Any],
    events: queue.Queue[TaskEvent],
    cancel_event: threading.Event,
) -> list[ProcessedAudio]:
    checkpoint = checkpoint.expanduser().resolve()
    output_directory = output_directory.expanduser().resolve()
    validated = validate_job(
        items, checkpoint, output_directory, settings, model_info
    )
    target_rate = int(model_info["pipeline_sample_rate"])
    for _, metadata, _ in validated:
        if metadata.sample_rate != target_rate:
            events.put(
                TaskEvent(
                    "warning",
                    f"{metadata.path.name}: enhancement.py remuestreará "
                    f"{metadata.sample_rate} Hz a {target_rate} Hz.",
                )
            )

    started_at = datetime.now(timezone.utc).isoformat()
    result_files: list[dict[str, str]] = []
    job_directory = Path(
        tempfile.mkdtemp(prefix="sgmse-job-", dir=prepare_runtime())
    )
    staged_input = job_directory / "input"
    staged_output = job_directory / "output"
    process: subprocess.Popen[str] | None = None
    reader: threading.Thread | None = None
    staged_names: list[Path] = []
    try:
        staged_input.mkdir()
        staged_output.mkdir()
        for index, (item, _, _) in enumerate(validated):
            if cancel_event.is_set():
                raise TaskCancelled("Procesamiento cancelado durante la preparación.")
            staged_name = Path(f"audio_{index:06d}{item.source.suffix.lower()}")
            shutil.copyfile(item.source, staged_input / staged_name)
            staged_names.append(staged_name)

        arguments = build_enhancement_arguments(
            staged_input, staged_output, checkpoint, settings
        )
        command = launch_command(["--sgmse-worker", *arguments])
        events.put(TaskEvent("status", "Ejecutando enhancement.py…"))
        process = subprocess.Popen(
            command,
            cwd=project_root(),
            env=subprocess_environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=worker_creationflags(),
        )
        if process.stdout is None:
            raise RuntimeError("No se pudo capturar la salida de enhancement.py.")

        output_lines: list[str] = []
        output_lock = threading.Lock()

        def capture_output() -> None:
            assert process is not None and process.stdout is not None
            for line in process.stdout:
                message = line.rstrip()
                if message:
                    with output_lock:
                        output_lines.append(message)
                        if len(output_lines) > 500:
                            del output_lines[:100]
                    events.put(TaskEvent("log", message))

        reader = threading.Thread(target=capture_output, daemon=True)
        reader.start()
        stable_sizes: dict[Path, tuple[int, int]] = {}
        completed_paths: set[Path] = set()
        while process.poll() is None:
            if cancel_event.is_set():
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise TaskCancelled("Procesamiento cancelado.")
            for staged_name in staged_names:
                result_path = staged_output / staged_name
                if not result_path.is_file() or result_path in completed_paths:
                    continue
                size = result_path.stat().st_size
                previous = stable_sizes.get(result_path)
                stable_count = previous[1] + 1 if previous and previous[0] == size else 0
                stable_sizes[result_path] = (size, stable_count)
                if size > 0 and stable_count >= 1:
                    completed_paths.add(result_path)
                    events.put(
                        TaskEvent(
                            "progress",
                            result_path.name,
                            len(completed_paths),
                            len(staged_names),
                        )
                    )
            time.sleep(0.25)
        return_code = process.wait()
        reader.join()
        if return_code:
            with output_lock:
                detail = "\n".join(output_lines[-25:])
            raise RuntimeError(
                f"enhancement.py terminó con código {return_code}."
                + (f"\n{detail}" if detail else "")
            )

        if cancel_event.is_set():
            raise TaskCancelled("Procesamiento cancelado antes de publicar resultados.")

        events.put(
            TaskEvent("progress", "Inferencia completada", len(staged_names), len(staged_names))
        )
        events.put(TaskEvent("publishing", "Publicando resultados…"))
        processed: list[ProcessedAudio] = []
        for (item, source_metadata, destination), staged_name in zip(
            validated, staged_names
        ):
            staged_result = staged_output / staged_name
            if not staged_result.is_file() or staged_result.stat().st_size == 0:
                raise RuntimeError(
                    f"enhancement.py no generó un resultado válido para {item.source}."
                )
            result_metadata = inspect_audio(staged_result)
            if result_metadata.sample_rate != target_rate:
                raise RuntimeError(
                    f"Frecuencia inesperada en {staged_result.name}: "
                    f"{result_metadata.sample_rate} Hz; se esperaba {target_rate} Hz."
                )
            if result_metadata.channels != 1:
                raise RuntimeError(
                    f"El resultado de {staged_result.name} tiene "
                    f"{result_metadata.channels} canales; se esperaba audio mono."
                )
            duration_tolerance = 2 / target_rate
            if (
                abs(result_metadata.duration_seconds - source_metadata.duration_seconds)
                > duration_tolerance
            ):
                raise RuntimeError(
                    f"La duración de {staged_result.name} no coincide con la entrada "
                    f"({result_metadata.duration_seconds:.6f} s frente a "
                    f"{source_metadata.duration_seconds:.6f} s)."
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise FileExistsError(
                    f"El resultado apareció durante el proceso y no se sobrescribirá: "
                    f"{destination}"
                )
            partial = destination.with_name(
                f".{destination.name}.{uuid.uuid4().hex}.partial"
            )
            try:
                shutil.copy2(staged_result, partial)
                os.rename(partial, destination)
            finally:
                if partial.exists():
                    partial.unlink()
            processed.append(
                ProcessedAudio(item.source, destination, inspect_audio(destination))
            )
            result_files.append(
                {"input": str(item.source), "output": str(destination)}
            )
            events.put(TaskEvent("status", f"Publicado {destination.name}"))

        _record(
            "completed",
            checkpoint,
            output_directory,
            settings,
            started_at,
            result_files,
        )
        return processed
    except TaskCancelled as error:
        _record(
            "cancelled",
            checkpoint,
            output_directory,
            settings,
            started_at,
            result_files,
            str(error),
        )
        raise
    except Exception as error:
        _record(
            "failed",
            checkpoint,
            output_directory,
            settings,
            started_at,
            result_files,
            str(error),
        )
        raise
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if reader is not None:
            reader.join(timeout=5)
        if process is not None and process.stdout is not None:
            process.stdout.close()
        shutil.rmtree(job_directory)
