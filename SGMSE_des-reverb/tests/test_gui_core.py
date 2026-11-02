import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import unittest
import wave
from unittest.mock import Mock, patch
from pathlib import Path

from sgmse_gui.audio import (
    AudioItem,
    audio_items_from_paths,
    inspect_audio,
    validate_output_names,
)
from sgmse_gui.jobs import (
    InferenceSettings,
    TaskEvent,
    build_enhancement_arguments,
    run_inference,
    validate_job,
    validate_settings,
)
from sgmse_gui.app import SGMSEApp
from sgmse_gui.runtime import (
    launch_command,
    prepare_runtime,
    subprocess_environment,
    worker_creationflags,
)


class AudioMetadataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(dir=prepare_runtime())
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_wav(
        self, path: Path, channels: int = 1, sample_rate: int = 48000
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as audio:
            audio.setnchannels(channels)
            audio.setsampwidth(2)
            audio.setframerate(sample_rate)
            audio.writeframes(b"\0\0" * channels * 96)

    def test_reads_wav_metadata_without_changing_samples(self) -> None:
        audio_path = self.root / "sample.wav"
        self._write_wav(audio_path, sample_rate=48000)

        metadata = inspect_audio(audio_path)

        self.assertEqual(metadata.sample_rate, 48000)
        self.assertEqual(metadata.channels, 1)
        self.assertEqual(metadata.frames, 96)
        self.assertEqual(metadata.subtype, "PCM 16-bit")

    def test_folder_selection_keeps_relative_paths_and_deduplicates(self) -> None:
        folder = self.root / "input"
        self._write_wav(folder / "nested" / "sample.wav")
        self._write_wav(folder / "other.wav")

        items = audio_items_from_paths([folder, folder / "other.wav"])

        self.assertEqual(len(items), 2)
        relative_paths = {item.relative_output for item in items}
        self.assertEqual(
            relative_paths,
            {Path("nested") / "sample.wav", Path("other.wav")},
        )

    def test_output_name_collision_is_rejected_case_insensitively(self) -> None:
        first = self.root / "one" / "voice.wav"
        second = self.root / "two" / "VOICE.WAV"
        items = [
            AudioItem(first, Path("voice.wav")),
            AudioItem(second, Path("VOICE.WAV")),
        ]

        with self.assertRaisesRegex(ValueError, "mismo archivo de salida"):
            validate_output_names(items)

    def test_multichannel_audio_is_not_silently_downmixed(self) -> None:
        checkpoint = self.root / "model.ckpt"
        checkpoint.write_bytes(b"checkpoint")
        stereo = self.root / "stereo.wav"
        self._write_wav(stereo, channels=2)
        item = AudioItem(stereo, Path(stereo.name))
        model_info = {
            "checkpoint_path": str(checkpoint.resolve()),
            "score_model_sr": 16000,
            "supports_inference": True,
            "sde": "OUVESDE",
            "pipeline_sample_rate": 48000,
            "cuda_devices": [],
        }

        with self.assertRaisesRegex(ValueError, "no se mezclarán"):
            validate_job(
                [item],
                checkpoint,
                self.root / "output",
                InferenceSettings(device="cpu"),
                model_info,
            )

    def test_source_sample_rate_is_only_inspected(self) -> None:
        checkpoint = self.root / "model.ckpt"
        checkpoint.write_bytes(b"checkpoint")
        source = self.root / "source.wav"
        self._write_wav(source, sample_rate=16000)
        item = AudioItem(source, Path(source.name))
        model_info = {
            "checkpoint_path": str(checkpoint.resolve()),
            "score_model_sr": 16000,
            "supports_inference": True,
            "sde": "OUVESDE",
            "pipeline_sample_rate": 48000,
            "cuda_devices": [],
        }

        validated = validate_job(
            [item],
            checkpoint,
            self.root / "output",
            InferenceSettings(device="cpu"),
            model_info,
        )

        self.assertEqual(validated[0][1].sample_rate, 16000)
        self.assertFalse((self.root / "output").exists())

    def test_directml_and_hybrid_accept_compatible_48khz_checkpoints(self) -> None:
        checkpoint = self.root / "ears_reverb.ckpt"
        checkpoint.write_bytes(b"checkpoint")
        source = self.root / "mono.wav"
        self._write_wav(source, sample_rate=48000)
        item = AudioItem(source, Path(source.name))
        model_info = {
            "checkpoint_path": str(checkpoint.resolve()),
            "backbone": "ncsnpp_48k",
            "score_model_sr": 48000,
            "supports_inference": True,
            "sde": "OUVESDE",
            "pipeline_sample_rate": 48000,
            "cuda_devices": [],
            "directml_devices": ["AMD Radeon RX 6600"],
        }

        for device in ("directml", "hybrid"):
            with self.subTest(device=device):
                validated = validate_job(
                    [item],
                    checkpoint,
                    self.root / f"output-{device}",
                    InferenceSettings(device=device),
                    model_info,
                )
                self.assertEqual(len(validated), 1)

    def test_directml_rejects_missing_device_and_incompatible_backbone(self) -> None:
        checkpoint = self.root / "model.ckpt"
        checkpoint.write_bytes(b"checkpoint")
        source = self.root / "mono.wav"
        self._write_wav(source)
        item = AudioItem(source, Path(source.name))
        model_info = {
            "checkpoint_path": str(checkpoint.resolve()),
            "backbone": "ncsnpp_48k",
            "supports_inference": True,
            "sde": "OUVESDE",
            "pipeline_sample_rate": 48000,
            "cuda_devices": [],
            "directml_devices": [],
        }

        with self.assertRaisesRegex(ValueError, "DirectML no está disponible"):
            validate_job(
                [item],
                checkpoint,
                self.root / "missing-device",
                InferenceSettings(device="directml"),
                model_info,
            )

        model_info["directml_devices"] = ["AMD Radeon RX 6600"]
        model_info["backbone"] = "ncsnpp_v2"
        with self.assertRaisesRegex(ValueError, "solo para ncsnpp_48k"):
            validate_job(
                [item],
                checkpoint,
                self.root / "wrong-backbone",
                InferenceSettings(device="hybrid"),
                model_info,
            )


class InferenceConfigurationTests(unittest.TestCase):
    def test_inspected_compatible_checkpoint_offers_directml_and_hybrid(self) -> None:
        app = SGMSEApp.__new__(SGMSEApp)
        app.model_info = None
        app.inspected_checkpoint = ""
        app.model_info_text = Mock()
        app.model_status_var = Mock()
        app.device_var = Mock()
        app.device_var.get.return_value = "cpu"
        app.advanced_widgets = {"Dispositivo": Mock()}
        app._sync_settings_for_model = Mock()
        information = {
            "checkpoint_path": "ears_reverb.ckpt",
            "checkpoint_name": "ears_reverb.ckpt",
            "backbone": "ncsnpp_48k",
            "sde": "OUVESDE",
            "score_model_sr": 16000,
            "pipeline_sample_rate": 48000,
            "n_fft": 1534,
            "hop_length": 384,
            "transform_type": "exponent",
            "spec_factor": 0.065,
            "spec_abs_exponent": 0.667,
            "time_epsilon": 0.03,
            "sde_steps": 30,
            "cuda_devices": [],
            "directml_devices": ["AMD Radeon RX 6600"],
        }

        with patch.object(SGMSEApp, "_set_text"):
            app._show_model_info(information)

        app.advanced_widgets["Dispositivo"].configure.assert_called_once_with(
            values=["cpu", "directml", "hybrid"]
        )
        app.device_var.set.assert_not_called()

    def test_numba_cache_uses_a_short_directory_inside_runtime_temp(self) -> None:
        temporary = tempfile.TemporaryDirectory(dir=prepare_runtime())
        self.addCleanup(temporary.cleanup)
        runtime_temp = Path(temporary.name)

        with patch("sgmse_gui.runtime.prepare_runtime", return_value=runtime_temp):
            environment = subprocess_environment()

        cache_path = Path(environment["NUMBA_CACHE_DIR"])
        self.assertEqual(cache_path, runtime_temp / "n")
        self.assertTrue(cache_path.is_dir())

    def test_frozen_subprocesses_use_adjacent_worker_executable(self) -> None:
        executable = Path(r"F:\portable\SGMSE.exe")
        with (
            patch.object(sys, "frozen", True, create=True),
            patch.object(sys, "executable", str(executable)),
            patch("sgmse_gui.runtime.Path.is_file", return_value=True),
        ):
            command = launch_command(["--inspect-checkpoint", "model.ckpt"])

        self.assertEqual(
            command,
            [r"F:\portable\SGMSE-worker.exe", "--inspect-checkpoint", "model.ckpt"],
        )

    @unittest.skipUnless(os.name == "nt", "The packaged worker is Windows-only.")
    def test_frozen_worker_does_not_open_a_console_window(self) -> None:
        with patch.object(sys, "frozen", True, create=True):
            self.assertEqual(worker_creationflags(), subprocess.CREATE_NO_WINDOW)

    def test_builds_arguments_for_the_original_entry_point(self) -> None:
        settings = InferenceSettings(
            sampler_type="ode",
            corrector="none",
            corrector_steps=0,
            snr=0.25,
            steps=40,
            device="cpu",
            time_epsilon=0.02,
        )

        arguments = build_enhancement_arguments(
            Path(r"F:\temp\input"),
            Path(r"F:\temp\output"),
            Path(r"F:\models\ears.ckpt"),
            settings,
        )

        self.assertEqual(
            arguments,
            [
                "--test_dir", r"F:\temp\input",
                "--enhanced_dir", r"F:\temp\output",
                "--ckpt", r"F:\models\ears.ckpt",
                "--sampler_type", "ode",
                "--corrector", "none",
                "--corrector_steps", "0",
                "--snr", "0.25",
                "--N", "40",
                "--device", "cpu",
                "--t_eps", "0.02",
            ],
        )

    def test_rejects_invalid_original_cli_parameters(self) -> None:
        with self.assertRaisesRegex(ValueError, "al menos 1"):
            validate_settings(InferenceSettings(steps=0))
        with self.assertRaisesRegex(ValueError, "entre 0 y 1"):
            validate_settings(InferenceSettings(time_epsilon=1.0))

    def test_forwards_optional_random_seed(self) -> None:
        arguments = build_enhancement_arguments(
            Path(r"F:\temp\input"),
            Path(r"F:\temp\output"),
            Path(r"F:\models\ears.ckpt"),
            InferenceSettings(seed=2026),
        )

        self.assertEqual(arguments[-2:], ["--seed", "2026"])

    def test_windows_relative_output_stays_under_selected_directory(self) -> None:
        source = r"F:\audio\entrada\voz.wav"
        input_directory = r"F:\audio\entrada"
        output_directory = r"F:\audio\salida"

        relative_name = os.path.relpath(source, input_directory)
        output_path = os.path.join(output_directory, relative_name)

        self.assertEqual(relative_name, "voz.wav")
        self.assertEqual(output_path, r"F:\audio\salida\voz.wav")

    def test_batch_coordinator_stages_and_publishes_files(self) -> None:
        temporary = tempfile.TemporaryDirectory(dir=prepare_runtime())
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        source_a = root / "source-a" / "first.wav"
        source_b = root / "source-b" / "second.wav"
        for path in (source_a, source_b):
            path.parent.mkdir(parents=True)
            with wave.open(str(path), "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(48000)
                audio.writeframes(b"\0\0" * 24)

        checkpoint = root / "model.ckpt"
        checkpoint.write_bytes(b"checkpoint")
        output = root / "output"
        items = [
            AudioItem(source_a, Path("part-a") / "first.wav"),
            AudioItem(source_b, Path("part-b") / "second.wav"),
        ]
        model_info = {
            "checkpoint_path": str(checkpoint.resolve()),
            "score_model_sr": 16000,
            "supports_inference": True,
            "sde": "OUVESDE",
            "pipeline_sample_rate": 48000,
            "cuda_devices": [],
        }
        fake_worker = (
            "import argparse, pathlib, shutil, sys\n"
            "parser = argparse.ArgumentParser()\n"
            "parser.add_argument('--test_dir')\n"
            "parser.add_argument('--enhanced_dir')\n"
            "args, _ = parser.parse_known_args(sys.argv[2:])\n"
            "source = pathlib.Path(args.test_dir)\n"
            "target = pathlib.Path(args.enhanced_dir)\n"
            "target.mkdir(parents=True, exist_ok=True)\n"
            "for item in source.iterdir():\n"
            "    shutil.copyfile(item, target / item.name)\n"
        )
        events: queue.Queue[TaskEvent] = queue.Queue()
        with (
            patch("sgmse_gui.jobs.launch_command") as launch,
            patch("sgmse_gui.jobs.prepare_runtime", return_value=root / "temp"),
            patch("sgmse_gui.history.prepare_runtime", return_value=root / "temp"),
        ):
            (root / "temp").mkdir()
            launch.side_effect = lambda arguments: [
                sys.executable,
                "-c",
                fake_worker,
                *arguments,
            ]
            processed = run_inference(
                items,
                checkpoint,
                output,
                InferenceSettings(device="cpu"),
                model_info,
                events,
                threading.Event(),
            )

        self.assertEqual(len(processed), 2)
        self.assertEqual(
            {result.output.relative_to(output) for result in processed},
            {Path("part-a") / "first.wav", Path("part-b") / "second.wav"},
        )
        for result in processed:
            self.assertEqual(result.output.read_bytes(), result.source.read_bytes())
            self.assertEqual(result.metadata.sample_rate, 48000)
        history = json.loads((root / "temp" / "history.jsonl").read_text().splitlines()[0])
        self.assertEqual(history["status"], "completed")
        self.assertEqual(len(history["files"]), 2)


if __name__ == "__main__":
    unittest.main()
