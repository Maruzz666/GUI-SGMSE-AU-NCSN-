import wave
from dataclasses import dataclass
from pathlib import Path


SUPPORTED_EXTENSIONS = {".wav", ".flac"}


@dataclass(frozen=True)
class AudioItem:
    source: Path
    relative_output: Path


@dataclass(frozen=True)
class AudioMetadata:
    path: Path
    sample_rate: int
    channels: int
    frames: int
    duration_seconds: float
    subtype: str


def audio_items_from_paths(paths: list[Path]) -> list[AudioItem]:
    items: dict[Path, AudioItem] = {}
    for selected in paths:
        selected = selected.expanduser().resolve()
        if selected.is_file():
            if selected.suffix.lower() not in SUPPORTED_EXTENSIONS:
                raise ValueError(f"Formato no compatible: {selected.name}. Usa WAV o FLAC.")
            items[selected] = AudioItem(selected, Path(selected.name))
            continue
        if not selected.is_dir():
            raise FileNotFoundError(f"No existe la ruta seleccionada: {selected}")
        for audio_path in sorted(selected.rglob("*")):
            if audio_path.is_file() and audio_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                resolved = audio_path.resolve()
                items[resolved] = AudioItem(resolved, audio_path.relative_to(selected))
    if not items:
        raise ValueError("No se encontraron archivos WAV o FLAC.")
    return list(items.values())


def inspect_audio(path: Path) -> AudioMetadata:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"No existe el archivo de audio: {path}")
    extension = path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Formato no compatible: {path.name}. Usa WAV o FLAC.")

    if extension == ".wav":
        try:
            with wave.open(str(path), "rb") as audio:
                channels = audio.getnchannels()
                sample_rate = audio.getframerate()
                frames = audio.getnframes()
                subtype = f"PCM {audio.getsampwidth() * 8}-bit"
        except (wave.Error, EOFError):
            channels, sample_rate, frames, subtype = _soundfile_metadata(path)
    else:
        channels, sample_rate, frames, subtype = _soundfile_metadata(path)

    if sample_rate <= 0 or channels <= 0 or frames < 0:
        raise ValueError(f"Metadatos de audio no válidos: {path}")
    return AudioMetadata(
        path=path,
        sample_rate=sample_rate,
        channels=channels,
        frames=frames,
        duration_seconds=frames / sample_rate,
        subtype=subtype,
    )


def _soundfile_metadata(path: Path) -> tuple[int, int, int, str]:
    try:
        import soundfile
    except ImportError as error:
        raise RuntimeError(
            "Para inspeccionar FLAC instala las dependencias de audio del proyecto "
            "(soundfile/librosa). El archivo no se ha convertido."
        ) from error
    information = soundfile.info(str(path))
    return (
        information.channels,
        information.samplerate,
        information.frames,
        information.subtype,
    )


def validate_output_names(items: list[AudioItem]) -> None:
    names: dict[str, Path] = {}
    for item in items:
        relative = item.relative_output
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Ruta de salida no válida: {relative}")
        key = relative.as_posix().casefold()
        previous = names.get(key)
        if previous is not None and previous != item.source:
            raise ValueError(
                f"Dos entradas producirían el mismo archivo de salida: {relative}"
            )
        names[key] = item.source
