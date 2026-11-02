# SGMSE desktop GUI

The GUI is an adapter around the repository's existing `enhancement.py` entry
point. It does not implement the SGMSE model, STFT/ISTFT, spectral transforms,
padding, normalization, or samplers. The checkpoint is loaded using
`ScoreModel.load_from_checkpoint`, and every job runs the original enhancement
pipeline in a separate process.

## Run

From the repository directory, run:

```powershell
..\..\Scripts\python.exe -m sgmse_gui
```

Install the project's inference requirements from `requirements.txt` first.
The GUI itself uses the Python standard-library Tk interface and adds no
runtime GUI dependency.
When the bundled default checkpoint is available, the GUI inspects it at
startup; choosing another checkpoint from the model list or file picker also
starts inspection. GPU choices appear in the advanced device selector only
after a compatible checkpoint has been inspected. The selected backend remains
CPU until the user explicitly chooses another device.

## Audio handling

- WAV and FLAC files can be selected individually, in batches, or recursively
  from a folder. Folder-relative paths are preserved in the results.
- The GUI reads metadata separately and stages input files as byte-for-byte
  copies. It does not decode, normalize, remix, resample, or rewrite the audio
  passed to `enhancement.py`.
- The existing pipeline chooses 48 kHz for `ncsnpp_48k` and 16 kHz otherwise.
  When input and target rates differ, the GUI warns that the original pipeline
  will apply its existing `librosa.resample` step.
- The displayed `ScoreModel.sr` is the checkpoint's metric setting, not the
  enhancement input rate. Inference rate is selected from the backbone exactly
  as it is in `enhancement.py`.
- The current backbones consume mono speech. Multichannel inputs are rejected
  before inference; channels are never discarded or averaged.
- Original and enhanced WAV files can be played directly on Windows. FLAC is
  accepted for inference but is not decoded for integrated playback.
- Results are never overwritten. Choose another output folder if a destination
  file already exists.

The model inspection panel reports checkpoint fields, including backbone,
SDE, sampling rate, STFT parameters, and available CUDA devices. For `SBVESDE`,
the original enhancement entry point uses CUDA directly and does not apply the
GUI's PC corrector or step-count controls; the interface disables those
controls and requires CUDA.

## Portable runtime data

The GUI stores temporary job data, history, and logs in `.tmp` next to the
repository while running from source, or next to the executable when frozen.
It does not use `%APPDATA%` for these files. The Numba cache uses a short
`.tmp\n` path so generated cache filenames stay within Windows path limits.
Canceling a job terminates its inference process and discards staged outputs;
completed outputs are published only after the original pipeline succeeds.

## AMD DirectML inference

The AMD runtime is deliberately isolated from the existing CPU environment.
From the repository directory, create/use `.venv\gpu-directml` and install the
separate runtime requirements:

```powershell
..\..\..\Scripts\python.exe -m venv ..\..\..\gpu-directml
..\..\..\gpu-directml\Scripts\python.exe -m pip install -r requirements-directml.txt
..\..\..\gpu-directml\Scripts\python.exe -m sgmse_gui
```

The installed GUI detects DirectML adapters after inspecting a checkpoint.
The `directml` and `hybrid` choices are offered only for the compatible
`ncsnpp_48k` / `OUVESDE` checkpoint. Both leave the complex STFT, SDE sampling,
and audio reconstruction on CPU because DirectML does not support PyTorch's
`ComplexFloat` operations. The neural network's four real-valued channels run
on the AMD GPU. `directml` also runs the FIR up/downsampling kernels on the
GPU; `hybrid` executes those FIR operations on CPU and copies their results
back to the GPU. Neither mode changes model weights, sample rate, or network
architecture. Stereo files remain rejected.
Both modes still need enough GPU memory for the neural network; `hybrid` is not
a VRAM fallback. A DirectML out-of-memory error is reported to the user rather
than silently switching devices; close other GPU-heavy applications or select
CPU if the adapter lacks free memory.

The original FIR helper dispatches to a CUDA extension on non-CPU devices, but
that extension is unavailable in the CPU Torch build required by
`torch-directml`; its CPU reference also creates a six-dimensional intermediate,
above DirectML's tensor-rank limit. CPU, DirectML, and hybrid use the same FIR
math with rank-four operations and bounded convolution batches. The checkpoint's
EMA weights are applied on CPU before only the neural network is transferred,
avoiding a second GPU copy of EMA bookkeeping.
For DirectML and hybrid, saved U-Net skip activations are kept on CPU and copied
back to the compute device when needed. This reduces GPU-resident activations
without changing model weights or network layers.

The portable build uses the isolated DirectML environment but also includes
the CPU Torch path. CPU, DirectML, and hybrid therefore run from the same
distribution; the original CPU-only distribution is replaced when rebuilding.

### Packaged DirectML validation

The portable build was run on an AMD Radeon RX 6600 with
`torch-directml==0.2.5.dev240914`, Torch 2.4.1, and the original
`ears_reverb.ckpt`. The GUI executable opened successfully. Audio jobs used the
packaged worker and original enhancement pipeline with `N=1`, no corrector,
and seed 41. Timings are end-to-end wall-clock measurements from one run.

| Backend | Input | Result | Total process time |
|---|---|---|---:|
| CPU | Full `Audio_prueba.wav` (6.594458 s) | Passed; finite mono PCM-16, 48 kHz, 316,534 frames | 38.73 s |
| CPU | 0.4 s segment | Passed; finite mono PCM-16, 48 kHz, 19,200 frames | 9.74 s |
| DirectML | 0.4 s segment | Passed; finite mono PCM-16, 48 kHz, 19,200 frames | 11.20 s |
| Hybrid | 0.4 s segment | Passed; finite mono PCM-16, 48 kHz, 19,200 frames | 11.02 s |

For the 0.4-second test, CPU vs. DirectML output had MAE `6.23e-7`, RMSE
`4.36e-6`, maximum absolute difference `3.05e-5`, and relative L2 error
`4.53e-6`. CPU vs. hybrid had MAE `6.07e-7`, RMSE `4.30e-6`, maximum
absolute difference `3.05e-5`, and relative L2 error `4.47e-6`. The DirectML
and hybrid `upfirdn2d` outputs also matched the CPU reference exactly for
tested upsampling/downsampling shapes. These timings are not a performance
guarantee.

**Full-file GPU processing is not validated.** On this machine, running the
full 6.594458-second WAV in either DirectML or hybrid emitted DirectML allocator
out-of-memory warnings and terminated the packaged worker with Windows exit
code `0xC0000005` (`-1073741819`); neither run produced an output WAV. The GUI
records a nonzero worker exit as a failed job and does not silently switch to
CPU. Use CPU for this full-length file; DirectML and hybrid have only been
verified with the short segment above. Do not treat either GPU mode as
full-file-capable until the long-input test passes on the target machine.

The isolated operator probe passed the real-valued convolutions, normalization,
activations, interpolation, FIR `upfirdn2d`, and the tested arithmetic
operations. `view_as_complex`, FFT, and STFT fail on DirectML with the native
`ComplexFloat` unsupported-type error; those operations intentionally remain
on CPU in both backends.

## Portable Windows build with AMD support

Build from the repository directory using only `.venv\gpu-directml`. Preserve
the checkpoint outside the destination directory before PyInstaller replaces
`dist\SGMSE`:

```powershell
$buildPython = "..\..\..\gpu-directml\Scripts\python.exe"
$env:TEMP = Join-Path (Get-Location) ".tmp"
$env:TMP = $env:TEMP
$env:PYINSTALLER_CONFIG_DIR = Join-Path (Get-Location) ".tmp\pyinstaller-config"
& $buildPython -m pip install --no-cache-dir -r requirements-build.txt
New-Item -ItemType Directory -Force .tmp\build-input\models | Out-Null
Copy-Item dist\SGMSE\_internal\models\ears_reverb.ckpt .tmp\build-input\models\
& $buildPython -m PyInstaller --clean --noconfirm --distpath dist --workpath .tmp\pyinstaller-directml sgmse_gui.spec
```

The portable application is written to `dist\SGMSE`. Keep the whole directory
together when moving it. `SGMSE.exe` opens the interface and
`SGMSE-worker.exe` is the hidden-console helper used for checkpoint inspection
and inference. The bundled checkpoint and DirectML runtime are under the
application data directory. CPU processing does not require an AMD adapter;
DirectML and hybrid appear only when a compatible adapter and checkpoint are
available. Runtime logs, history, and temporary files remain in
`dist\SGMSE\.tmp`, beside the executable.

## Tests

Run the GUI adapter tests with:

```powershell
..\..\Scripts\python.exe -m unittest discover -s tests -v
```

These tests cover metadata inspection, batch path handling, output collisions,
channel compatibility, inference argument forwarding, and Windows output-path
semantics. They do not replace an end-to-end checkpoint inference test.
