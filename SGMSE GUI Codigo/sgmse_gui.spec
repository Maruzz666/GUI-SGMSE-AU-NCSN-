from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules
import librosa


ROOT = Path(SPECPATH)
datas = [(str(ROOT / "enhancement.py"), ".")]
binaries = []
hiddenimports = collect_submodules("sgmse")
hiddenimports.extend(collect_submodules("torch_directml"))
librosa_root = Path(librosa.__file__).parent
for source in librosa_root.rglob("*.py"):
    destination = Path("librosa") / source.relative_to(librosa_root).parent
    datas.append((str(source), str(destination)))

for package in ("librosa", "pytorch_lightning", "torchaudio", "torch_directml"):
    package_datas, package_binaries, package_hiddenimports = collect_all(package)
    datas.extend(package_datas)
    binaries.extend(package_binaries)
    hiddenimports.extend(package_hiddenimports)

checkpoint_root = ROOT / ".tmp" / "build-input" / "models"
if not checkpoint_root.is_dir():
    checkpoint_root = ROOT / "models"
checkpoints = sorted(checkpoint_root.rglob("*.ckpt")) if checkpoint_root.is_dir() else []
if not checkpoints:
    raise FileNotFoundError(
        "No se encontró ningún checkpoint. Coloca los .ckpt en models o "
        ".tmp\\build-input\\models antes de compilar."
    )

for checkpoint in checkpoints:
    destination = Path("models") / checkpoint.relative_to(checkpoint_root).parent
    datas.append((str(checkpoint), str(destination)))

analysis = Analysis(
    [str(ROOT / "sgmse_gui_entry.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(ROOT / "sgmse_gui" / "pyinstaller_librosa_source.py")],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(analysis.pure)

gui_executable = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="SGMSE",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
)

worker_executable = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="SGMSE-worker",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)

COLLECT(
    gui_executable,
    worker_executable,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    name="SGMSE",
)
