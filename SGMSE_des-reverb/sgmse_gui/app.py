import logging
import os
import queue
import subprocess
import sys
import threading
import traceback
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .audio import AudioItem, AudioMetadata, audio_items_from_paths, inspect_audio
from .checkpoint import inspect_checkpoint
from .history import read_history
from .jobs import (
    InferenceSettings,
    ProcessedAudio,
    TaskCancelled,
    TaskEvent,
    run_inference,
    validate_settings,
)
from .logging_store import create_logger
from .runtime import prepare_runtime, project_root


class SGMSEApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("SGMSE · Speech Enhancement")
        self.root.geometry("1180x820")
        self.root.minsize(980, 680)
        self.root.configure(background="#101827")

        prepare_runtime()
        self.logger = create_logger()
        self.events: queue.Queue[TaskEvent] = queue.Queue()
        self.items: dict[str, tuple[AudioItem, AudioMetadata]] = {}
        self.results: dict[str, ProcessedAudio] = {}
        self.model_info: dict | None = None
        self.inspected_checkpoint = ""
        self.scanning = False
        self.inspecting = False
        self.processing = False
        self.close_requested = False
        self.cancel_event = threading.Event()

        self.checkpoint_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.device_var = tk.StringVar(value="cpu")
        self.seed_var = tk.StringVar()
        self.sampler_var = tk.StringVar(value="pc")
        self.corrector_var = tk.StringVar(value="ald")
        self.corrector_steps_var = tk.StringVar(value="1")
        self.snr_var = tk.StringVar(value="0.5")
        self.steps_var = tk.StringVar(value="30")
        self.epsilon_var = tk.StringVar(value="0.03")
        self.advanced_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Listo")
        self.model_status_var = tk.StringVar(value="Selecciona e inspecciona un checkpoint.")

        self._configure_theme()
        self._build_ui()
        self.checkpoint_var.trace_add("write", self._checkpoint_changed)
        self._load_available_checkpoints()
        self._refresh_history()
        if Path(self.checkpoint_var.get().strip()).is_file():
            self.root.after_idle(self._inspect_model)
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.root.after(100, self._drain_events)
        self.logger.info("Interfaz SGMSE iniciada")

    def _configure_theme(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError as error:
            self.logger.warning("No se pudo aplicar el tema clam: %s", error)
        style.configure(
            "TFrame",
            background="#101827",
        )
        style.configure(
            "Panel.TFrame",
            background="#172337",
        )
        style.configure(
            "TLabel",
            background="#101827",
            foreground="#e7edf5",
            font=("Segoe UI", 10),
        )
        style.configure(
            "Title.TLabel",
            background="#101827",
            foreground="#f8fafc",
            font=("Segoe UI Semibold", 21),
        )
        style.configure(
            "Subtitle.TLabel",
            background="#101827",
            foreground="#a7b4c6",
            font=("Segoe UI", 10),
        )
        style.configure(
            "Panel.TLabel",
            background="#172337",
            foreground="#e7edf5",
        )
        style.configure("TButton", font=("Segoe UI", 9), padding=(10, 7))
        style.configure(
            "Accent.TButton",
            background="#32c48d",
            foreground="#071b18",
            font=("Segoe UI Semibold", 10),
            padding=(13, 9),
        )
        style.map(
            "Accent.TButton",
            background=[("disabled", "#526277"), ("active", "#57d9a5")],
        )
        style.configure(
            "Treeview",
            background="#172337",
            fieldbackground="#172337",
            foreground="#e7edf5",
            rowheight=27,
            borderwidth=0,
        )
        style.configure(
            "Treeview.Heading",
            background="#24344d",
            foreground="#e7edf5",
            font=("Segoe UI Semibold", 9),
        )
        style.map(
            "Treeview",
            background=[("selected", "#246c62")],
            foreground=[("selected", "#ffffff")],
        )
        style.configure(
            "TNotebook",
            background="#101827",
            borderwidth=0,
        )
        style.configure(
            "TNotebook.Tab",
            background="#24344d",
            foreground="#dce5f1",
            padding=(14, 8),
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", "#32c48d")],
            foreground=[("selected", "#071b18")],
        )
        style.configure(
            "Horizontal.TProgressbar",
            troughcolor="#24344d",
            background="#32c48d",
        )

    def _build_ui(self) -> None:
        shell = ttk.Frame(self.root, padding=(22, 18))
        shell.pack(fill="both", expand=True)
        header = ttk.Frame(shell)
        header.pack(fill="x", pady=(0, 15))
        ttk.Label(header, text="SGMSE", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Enhancement y dereverberación mediante el pipeline científico original",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(3, 0))

        self.notebook = ttk.Notebook(shell)
        self.notebook.pack(fill="both", expand=True)
        self.process_tab = ttk.Frame(self.notebook, padding=14)
        self.models_tab = ttk.Frame(self.notebook, padding=14)
        self.history_tab = ttk.Frame(self.notebook, padding=14)
        self.notebook.add(self.process_tab, text="  Procesamiento  ")
        self.notebook.add(self.models_tab, text="  Modelos  ")
        self.notebook.add(self.history_tab, text="  Historial y eventos  ")
        self._build_process_tab()
        self._build_models_tab()
        self._build_history_tab()

        status_bar = ttk.Frame(shell)
        status_bar.pack(fill="x", pady=(12, 0))
        ttk.Label(status_bar, textvariable=self.status_var).pack(side="left")
        ttk.Label(
            status_bar,
            text=f"Temporales: {prepare_runtime()}",
            style="Subtitle.TLabel",
        ).pack(side="right")

    def _build_process_tab(self) -> None:
        tab = self.process_tab
        ttk.Label(tab, text="Archivos de entrada").pack(anchor="w")
        input_actions = ttk.Frame(tab)
        input_actions.pack(fill="x", pady=(6, 8))
        self.add_files_button = ttk.Button(
            input_actions, text="Añadir archivos", command=self._add_files
        )
        self.add_files_button.pack(side="left")
        self.add_folder_button = ttk.Button(
            input_actions, text="Añadir carpeta", command=self._add_folder
        )
        self.add_folder_button.pack(side="left", padx=(8, 0))
        self.clear_button = ttk.Button(
            input_actions, text="Vaciar lista", command=self._clear_inputs
        )
        self.clear_button.pack(side="left", padx=(8, 0))

        input_frame = ttk.Frame(tab)
        input_frame.pack(fill="both", expand=True)
        self.input_table = ttk.Treeview(
            input_frame,
            columns=("rate", "channels", "duration", "format"),
            show="tree headings",
            height=5,
        )
        self.input_table.heading("#0", text="Archivo")
        self.input_table.heading("rate", text="Frecuencia")
        self.input_table.heading("channels", text="Canales")
        self.input_table.heading("duration", text="Duración")
        self.input_table.heading("format", text="Formato")
        self.input_table.column("#0", width=400, anchor="w")
        self.input_table.column("rate", width=110, anchor="center")
        self.input_table.column("channels", width=85, anchor="center")
        self.input_table.column("duration", width=110, anchor="center")
        self.input_table.column("format", width=120, anchor="center")
        input_scroll = ttk.Scrollbar(
            input_frame, orient="vertical", command=self.input_table.yview
        )
        self.input_table.configure(yscrollcommand=input_scroll.set)
        self.input_table.pack(side="left", fill="both", expand=True)
        input_scroll.pack(side="right", fill="y")
        self.input_table.bind("<<TreeviewSelect>>", self._show_input_details)

        details = ttk.Frame(tab)
        details.pack(fill="x", pady=(7, 9))
        self.input_detail_var = tk.StringVar(
            value="Los metadatos se inspeccionan sin convertir las muestras."
        )
        ttk.Label(
            details,
            textvariable=self.input_detail_var,
            style="Subtitle.TLabel",
            wraplength=1000,
        ).pack(anchor="w")

        model_row = ttk.Frame(tab)
        model_row.pack(fill="x", pady=(4, 8))
        ttk.Label(model_row, text="Checkpoint").pack(side="left")
        self.checkpoint_entry = ttk.Entry(
            model_row, textvariable=self.checkpoint_var
        )
        self.checkpoint_entry.pack(side="left", fill="x", expand=True, padx=8)
        self.checkpoint_button = ttk.Button(
            model_row, text="Examinar…", command=self._browse_checkpoint
        )
        self.checkpoint_button.pack(side="left")
        self.inspect_button = ttk.Button(
            model_row, text="Inspeccionar", command=self._inspect_model
        )
        self.inspect_button.pack(side="left", padx=(8, 0))
        ttk.Label(
            tab,
            textvariable=self.model_status_var,
            style="Subtitle.TLabel",
            wraplength=1000,
        ).pack(anchor="w", pady=(0, 8))

        output_row = ttk.Frame(tab)
        output_row.pack(fill="x", pady=(0, 8))
        ttk.Label(output_row, text="Carpeta de resultados").pack(side="left")
        ttk.Entry(output_row, textvariable=self.output_var).pack(
            side="left", fill="x", expand=True, padx=8
        )
        self.output_button = ttk.Button(
            output_row, text="Elegir…", command=self._browse_output
        )
        self.output_button.pack(side="left")
        self.open_output_button = ttk.Button(
            output_row, text="Abrir", command=self._open_output
        )
        self.open_output_button.pack(side="left", padx=(8, 0))

        advanced_header = ttk.Frame(tab)
        advanced_header.pack(fill="x", pady=(3, 2))
        ttk.Checkbutton(
            advanced_header,
            text="Mostrar parámetros avanzados",
            variable=self.advanced_var,
            command=self._toggle_advanced,
        ).pack(side="left")
        self.advanced_panel = ttk.LabelFrame(
            tab, text="Parámetros que enhancement.py admite", padding=10
        )
        self._build_advanced_controls()

        result_header = ttk.Frame(tab)
        result_header.pack(fill="x", pady=(9, 5))
        ttk.Label(result_header, text="Resultados").pack(side="left")
        self.play_input_button = ttk.Button(
            result_header,
            text="Reproducir entrada",
            command=lambda: self._play_selected("input"),
        )
        self.play_input_button.pack(side="right")
        self.play_output_button = ttk.Button(
            result_header,
            text="Reproducir mejorado",
            command=lambda: self._play_selected("output"),
        )
        self.play_output_button.pack(side="right", padx=(0, 8))
        self.stop_playback_button = ttk.Button(
            result_header, text="Detener audio", command=self._stop_playback
        )
        self.stop_playback_button.pack(side="right", padx=(0, 8))
        self.results_table = ttk.Treeview(
            tab,
            columns=("rate", "channels", "duration", "output"),
            show="tree headings",
            height=3,
        )
        self.results_table.heading("#0", text="Entrada")
        self.results_table.heading("rate", text="Salida Hz")
        self.results_table.heading("channels", text="Canales")
        self.results_table.heading("duration", text="Duración")
        self.results_table.heading("output", text="Archivo generado")
        self.results_table.column("#0", width=250, anchor="w")
        self.results_table.column("rate", width=95, anchor="center")
        self.results_table.column("channels", width=80, anchor="center")
        self.results_table.column("duration", width=105, anchor="center")
        self.results_table.column("output", width=480, anchor="w")
        self.results_table.pack(fill="both", expand=True)

        task_controls = ttk.Frame(tab)
        task_controls.pack(fill="x", pady=(10, 0))
        self.progress = ttk.Progressbar(
            task_controls, mode="determinate", maximum=1, value=0
        )
        self.progress.pack(side="left", fill="x", expand=True, padx=(0, 12))
        self.start_button = ttk.Button(
            task_controls,
            text="Procesar",
            style="Accent.TButton",
            command=self._start_processing,
        )
        self.start_button.pack(side="left")
        self.cancel_button = ttk.Button(
            task_controls,
            text="Cancelar",
            command=self._cancel_processing,
            state="disabled",
        )
        self.cancel_button.pack(side="left", padx=(8, 0))

    def _build_advanced_controls(self) -> None:
        fields = [
            ("Sampler", self.sampler_var, ("pc", "ode")),
            ("Corrector", self.corrector_var, ("ald", "langevin", "none")),
            ("Pasos corrector", self.corrector_steps_var, None),
            ("SNR", self.snr_var, None),
            ("N", self.steps_var, None),
            ("t_eps", self.epsilon_var, None),
            ("Dispositivo", self.device_var, ("cpu",)),
            ("Semilla (opcional)", self.seed_var, None),
        ]
        self.advanced_widgets: dict[str, ttk.Widget] = {}
        for column, (label, variable, values) in enumerate(fields):
            group = ttk.Frame(self.advanced_panel)
            group.grid(row=0, column=column, sticky="ew", padx=(0, 10))
            ttk.Label(group, text=label).pack(anchor="w")
            if values is not None:
                widget = ttk.Combobox(
                    group, textvariable=variable, values=values, state="readonly",
                    width=13,
                )
                if label == "Dispositivo":
                    widget.configure(state="readonly")
                if label == "Sampler":
                    variable.trace_add("write", self._sampler_changed)
            else:
                widget = ttk.Entry(group, textvariable=variable, width=14)
            widget.pack(fill="x", pady=(4, 0))
            self.advanced_widgets[label] = widget
            self.advanced_panel.columnconfigure(column, weight=1)

    def _build_models_tab(self) -> None:
        tab = self.models_tab
        ttk.Label(tab, text="Checkpoints disponibles").pack(anchor="w")
        actions = ttk.Frame(tab)
        actions.pack(fill="x", pady=(7, 8))
        ttk.Button(
            actions, text="Actualizar lista", command=self._load_available_checkpoints
        ).pack(side="left")
        ttk.Button(
            actions,
            text="Elegir archivo…",
            command=self._browse_checkpoint,
        ).pack(side="left", padx=(8, 0))
        self.checkpoint_table = ttk.Treeview(
            tab,
            columns=("size", "modified"),
            show="tree headings",
            height=8,
        )
        self.checkpoint_table.heading("#0", text="Checkpoint")
        self.checkpoint_table.heading("size", text="Tamaño")
        self.checkpoint_table.heading("modified", text="Modificado")
        self.checkpoint_table.column("#0", width=650, anchor="w")
        self.checkpoint_table.column("size", width=120, anchor="center")
        self.checkpoint_table.column("modified", width=180, anchor="center")
        self.checkpoint_table.pack(fill="both", expand=True)
        self.checkpoint_table.bind(
            "<<TreeviewSelect>>", self._select_checkpoint_from_table
        )
        self.model_info_text = self._text_area(tab, height=10)
        self.model_info_text.pack(fill="both", expand=True, pady=(10, 0))

    def _build_history_tab(self) -> None:
        tab = self.history_tab
        history_actions = ttk.Frame(tab)
        history_actions.pack(fill="x", pady=(0, 8))
        ttk.Button(
            history_actions, text="Actualizar historial", command=self._refresh_history
        ).pack(side="left")
        self.history_table = ttk.Treeview(
            tab,
            columns=("status", "files", "checkpoint", "output"),
            show="tree headings",
            height=8,
        )
        self.history_table.heading("#0", text="Fecha")
        self.history_table.heading("status", text="Estado")
        self.history_table.heading("files", text="Archivos")
        self.history_table.heading("checkpoint", text="Checkpoint")
        self.history_table.heading("output", text="Resultados")
        self.history_table.column("#0", width=200, anchor="w")
        self.history_table.column("status", width=100, anchor="center")
        self.history_table.column("files", width=80, anchor="center")
        self.history_table.column("checkpoint", width=300, anchor="w")
        self.history_table.column("output", width=400, anchor="w")
        self.history_table.pack(fill="both", expand=True)
        ttk.Label(tab, text="Registro de eventos").pack(anchor="w", pady=(12, 5))
        self.log_text = self._text_area(tab, height=10)
        self.log_text.pack(fill="both", expand=True)

    @staticmethod
    def _text_area(parent: tk.Misc, height: int) -> tk.Text:
        text = tk.Text(
            parent,
            height=height,
            background="#172337",
            foreground="#e7edf5",
            insertbackground="#ffffff",
            relief="flat",
            wrap="word",
            padx=10,
            pady=8,
            font=("Consolas", 9),
        )
        return text

    def _toggle_advanced(self) -> None:
        if self.advanced_var.get():
            self.advanced_panel.pack(fill="x", pady=(4, 0))
        else:
            self.advanced_panel.pack_forget()

    def _add_files(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Seleccionar audios",
            filetypes=[
                ("Audio WAV/FLAC", "*.wav *.flac"),
                ("Todos los archivos", "*.*"),
            ],
        )
        if paths:
            self._scan_inputs([Path(path) for path in paths])

    def _add_folder(self) -> None:
        selected = filedialog.askdirectory(title="Seleccionar carpeta de audios")
        if selected:
            self._scan_inputs([Path(selected)])

    def _scan_inputs(self, paths: list[Path]) -> None:
        if self.scanning or self.processing:
            return
        self.scanning = True
        self.status_var.set("Inspeccionando archivos…")

        def scan() -> None:
            try:
                selections = audio_items_from_paths(paths)
                for item in selections:
                    try:
                        metadata = inspect_audio(item.source)
                    except (OSError, RuntimeError, ValueError) as error:
                        self.events.put(
                            TaskEvent(
                                "scan_error",
                                f"{item.source}: {error}",
                            )
                        )
                    else:
                        self.events.put(
                            TaskEvent("scan_item", payload=(item, metadata))
                        )
            except Exception as error:
                self.logger.exception("No se pudieron inspeccionar las entradas")
                self.events.put(TaskEvent("scan_failed", str(error)))
            finally:
                self.events.put(TaskEvent("scan_finished"))

        threading.Thread(target=scan, daemon=True).start()

    def _add_scanned_item(
        self, item: AudioItem, metadata: AudioMetadata
    ) -> None:
        source_key = str(item.source)
        if source_key in self.items:
            return
        destination_key = item.relative_output.as_posix().casefold()
        for existing, _ in self.items.values():
            if (
                existing.relative_output.as_posix().casefold() == destination_key
                and existing.source != item.source
            ):
                message = (
                    f"Se omitió {item.source}: su salida colisiona con "
                    f"{item.relative_output}."
                )
                self._log("warning", message)
                return
        self.items[source_key] = (item, metadata)
        self.input_table.insert(
            "",
            "end",
            iid=source_key,
            text=item.source.name,
            values=(
                f"{metadata.sample_rate:,} Hz",
                metadata.channels,
                f"{metadata.duration_seconds:.2f} s",
                metadata.subtype,
            ),
        )

    def _clear_inputs(self) -> None:
        if self.processing or self.scanning:
            return
        self.items.clear()
        self.input_table.delete(*self.input_table.get_children())
        self.input_detail_var.set("Lista vacía.")
        self.status_var.set("Listo")

    def _show_input_details(self, _event: tk.Event) -> None:
        selection = self.input_table.selection()
        if not selection:
            return
        entry = self.items.get(selection[0])
        if entry is None:
            return
        item, metadata = entry
        self.input_detail_var.set(
            f"{item.source}  ·  {metadata.sample_rate:,} Hz  ·  "
            f"{metadata.channels} canal(es)  ·  {metadata.frames:,} muestras "
            f"por canal  ·  {metadata.subtype}"
        )

    def _browse_checkpoint(self) -> None:
        selected = filedialog.askopenfilename(
            title="Seleccionar checkpoint SGMSE",
            filetypes=[("PyTorch checkpoints", "*.ckpt"), ("Todos", "*.*")],
            initialdir=str(project_root() / "models"),
        )
        if selected:
            self.checkpoint_var.set(selected)
            self.notebook.select(self.process_tab)
            self.root.after_idle(self._inspect_model)

    def _checkpoint_changed(self, *_args: str) -> None:
        current = self.checkpoint_var.get().strip()
        if current != self.inspected_checkpoint:
            self.model_info = None
            self.model_status_var.set(
                "Checkpoint pendiente de inspección con el cargador original."
            )
        self._sync_settings_for_model()

    def _load_available_checkpoints(self) -> None:
        if not hasattr(self, "checkpoint_table"):
            return
        self.checkpoint_table.delete(*self.checkpoint_table.get_children())
        model_directory = project_root() / "models"
        if not model_directory.is_dir():
            return
        checkpoints = sorted(model_directory.rglob("*.ckpt"))
        for checkpoint in checkpoints:
            stat = checkpoint.stat()
            self.checkpoint_table.insert(
                "",
                "end",
                iid=str(checkpoint.resolve()),
                text=str(checkpoint.resolve()),
                values=(
                    f"{stat.st_size / (1024 ** 3):.2f} GiB",
                    datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
                ),
            )
        default = model_directory / "ears_reverb.ckpt"
        if default.is_file() and not self.checkpoint_var.get():
            self.checkpoint_var.set(str(default.resolve()))

    def _select_checkpoint_from_table(self, _event: tk.Event) -> None:
        selection = self.checkpoint_table.selection()
        if selection:
            self.checkpoint_var.set(selection[0])
            self.notebook.select(self.process_tab)
            self.root.after_idle(self._inspect_model)

    def _inspect_model(self) -> None:
        if self.inspecting or self.processing:
            return
        checkpoint = Path(self.checkpoint_var.get().strip())
        if not checkpoint.is_file():
            messagebox.showerror("Checkpoint", "Selecciona un checkpoint existente.")
            return
        self.inspecting = True
        self.inspect_button.configure(state="disabled")
        self.model_status_var.set("Cargando checkpoint con ScoreModel…")
        self._log("info", f"Inspeccionando checkpoint: {checkpoint}")

        def inspect() -> None:
            try:
                information = inspect_checkpoint(checkpoint)
            except Exception as error:
                detail = traceback.format_exc()
                self.logger.error("Error al inspeccionar checkpoint: %s\n%s", error, detail)
                self.events.put(
                    TaskEvent("model_failed", str(error), payload=detail)
                )
            else:
                self.events.put(TaskEvent("model_loaded", payload=information))

        threading.Thread(target=inspect, daemon=True).start()

    def _show_model_info(self, information: dict) -> None:
        self.model_info = information
        self.inspected_checkpoint = information["checkpoint_path"]
        details = [
            f"Checkpoint: {information['checkpoint_name']}",
            f"Backbone: {information['backbone']}",
            f"SDE: {information['sde']}",
            (
                f"ScoreModel.sr (métricas): {information['score_model_sr']:,} Hz"
                f"  ·  frecuencia de inferencia del backbone: "
                f"{information['pipeline_sample_rate']:,} Hz"
            ),
            (
                f"STFT del modelo: n_fft={information['n_fft']}, "
                f"hop_length={information['hop_length']}"
            ),
            (
                f"Transformación: {information['transform_type']}, "
                f"spec_factor={information['spec_factor']}, "
                f"spec_abs_exponent={information['spec_abs_exponent']}"
            ),
            (
                f"t_eps del checkpoint: {information['time_epsilon']}  ·  "
                f"pasos SDE: {information['sde_steps']}"
            ),
            f"CUDA: {', '.join(information['cuda_devices']) or 'no disponible'}",
            (
                "DirectML: "
                f"{', '.join(information.get('directml_devices', [])) or 'no disponible'}"
            ),
        ]
        if (
            information.get("directml_devices")
            and information["backbone"] == "ncsnpp_48k"
            and information["sde"] == "OUVESDE"
        ):
            details.extend(
                [
                    "DirectML: CNN de valores reales en GPU; STFT complejo y SDE en CPU.",
                    "Híbrido: además ejecuta up/downsampling FIR de la CNN en CPU.",
                    "Híbrido también requiere VRAM; si falta, selecciona CPU.",
                ]
            )
        self._set_text(self.model_info_text, "\n".join(details))
        self.model_status_var.set(
            f"Modelo cargado: {information['backbone']} / {information['sde']}"
        )
        cuda_devices = information["cuda_devices"]
        device_values = ["cpu"]
        if cuda_devices:
            device_values.extend(
                ["cuda"] if len(cuda_devices) == 1
                else [f"cuda:{index}" for index in range(len(cuda_devices))]
            )
        directml_devices = information.get("directml_devices", [])
        if (
            directml_devices
            and information["backbone"] == "ncsnpp_48k"
            and information["sde"] == "OUVESDE"
        ):
            device_values.extend(["directml", "hybrid"])
        device_widget = self.advanced_widgets["Dispositivo"]
        device_widget.configure(values=device_values)
        if self.device_var.get() not in device_values:
            self.device_var.set(device_values[-1] if cuda_devices else "cpu")
        self._sync_settings_for_model()

    def _sync_settings_for_model(self) -> None:
        if self.model_info is None:
            return
        is_sb = self.model_info["sde"] == "SBVESDE"
        sampler_is_pc = self.sampler_var.get() == "pc"
        corrector_state = "disabled" if is_sb or not sampler_is_pc else "readonly"
        numeric_corrector_state = "disabled" if is_sb or not sampler_is_pc else "normal"
        self.advanced_widgets["Corrector"].configure(state=corrector_state)
        self.advanced_widgets["Pasos corrector"].configure(
            state=numeric_corrector_state
        )
        self.advanced_widgets["SNR"].configure(state=numeric_corrector_state)
        self.advanced_widgets["N"].configure(
            state="disabled" if is_sb else "normal"
        )

    def _sampler_changed(self, *_args: str) -> None:
        self._sync_settings_for_model()

    def _browse_output(self) -> None:
        selected = filedialog.askdirectory(title="Seleccionar carpeta de resultados")
        if selected:
            self.output_var.set(selected)

    def _open_output(self) -> None:
        directory = Path(self.output_var.get().strip())
        if not directory.is_dir():
            messagebox.showerror("Resultados", "La carpeta seleccionada no existe.")
            return
        if sys.platform == "win32":
            os.startfile(str(directory))
        else:
            subprocess.Popen(["xdg-open", str(directory)])

    def _settings_from_ui(self) -> InferenceSettings:
        settings = InferenceSettings(
            sampler_type=self.sampler_var.get(),
            corrector=self.corrector_var.get(),
            corrector_steps=int(self.corrector_steps_var.get()),
            snr=float(self.snr_var.get()),
            steps=int(self.steps_var.get()),
            device=self.device_var.get(),
            time_epsilon=float(self.epsilon_var.get()),
            seed=(
                int(self.seed_var.get())
                if self.seed_var.get().strip()
                else None
            ),
        )
        validate_settings(settings)
        return settings

    def _start_processing(self) -> None:
        if self.processing or self.scanning or self.inspecting:
            return
        if not self.items:
            messagebox.showerror("Audio", "Añade al menos un archivo WAV o FLAC.")
            return
        checkpoint = Path(self.checkpoint_var.get().strip())
        if self.model_info is None or str(checkpoint.resolve()) != self.inspected_checkpoint:
            messagebox.showerror(
                "Modelo", "Inspecciona el checkpoint seleccionado antes de procesar."
            )
            return
        if not self.model_info.get("supports_inference", False):
            messagebox.showerror(
                "Modelo", "Este tipo de SDE no está soportado por enhancement.py."
            )
            return
        output_text = self.output_var.get().strip()
        if not output_text:
            self._browse_output()
            output_text = self.output_var.get().strip()
            if not output_text:
                return
        output_directory = Path(output_text)
        try:
            settings = self._settings_from_ui()
        except (ValueError, TypeError) as error:
            messagebox.showerror("Parámetros", str(error))
            return

        target_rate = int(self.model_info["pipeline_sample_rate"])
        mismatched = [
            metadata
            for _, metadata in self.items.values()
            if metadata.sample_rate != target_rate
        ]
        if mismatched:
            names = "\n".join(
                f"• {metadata.path.name}: {metadata.sample_rate:,} → "
                f"{target_rate:,} Hz"
                for metadata in mismatched[:8]
            )
            if len(mismatched) > 8:
                names += f"\n… y {len(mismatched) - 8} más"
            proceed = messagebox.askyesno(
                "Cambio de frecuencia del pipeline",
                f"El pipeline original enhancement.py aplicará su remuestreo "
                f"interno a {target_rate:,} Hz. La GUI no transforma las muestras.\n\n"
                f"{names}\n\n¿Continuar?",
            )
            if not proceed:
                return

        items = [entry[0] for entry in self.items.values()]
        model_info = dict(self.model_info)
        self.results.clear()
        self.results_table.delete(*self.results_table.get_children())
        self.cancel_event = threading.Event()
        self.processing = True
        self.start_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.progress.configure(maximum=len(items), value=0)
        self.status_var.set("Preparando lote…")
        self._log(
            "info",
            f"Iniciando {len(items)} archivo(s) con {checkpoint.name}; "
            f"salida: {output_directory}",
        )

        def process() -> None:
            try:
                completed = run_inference(
                    items,
                    checkpoint,
                    output_directory,
                    settings,
                    model_info,
                    self.events,
                    self.cancel_event,
                )
            except TaskCancelled as error:
                self.events.put(TaskEvent("task_cancelled", str(error)))
            except Exception as error:
                detail = traceback.format_exc()
                self.logger.error("Falló la tarea de inferencia: %s\n%s", error, detail)
                self.events.put(TaskEvent("task_failed", str(error), payload=detail))
            else:
                self.events.put(TaskEvent("task_completed", payload=completed))

        threading.Thread(target=process, daemon=True).start()

    def _cancel_processing(self) -> None:
        if not self.processing:
            return
        self.cancel_event.set()
        self.cancel_button.configure(state="disabled")
        self.status_var.set("Cancelando después de detener el proceso…")
        self._log("warning", "Se solicitó cancelar el lote")

    def _handle_event(self, event: TaskEvent) -> None:
        if event.kind == "scan_item":
            item, metadata = event.payload
            self._add_scanned_item(item, metadata)
        elif event.kind == "scan_error":
            self._log("error", event.message)
        elif event.kind == "scan_failed":
            self._log("error", event.message)
            messagebox.showerror("Inspección de audio", event.message)
        elif event.kind == "scan_finished":
            self.scanning = False
            self.status_var.set(f"{len(self.items)} archivo(s) listos")
        elif event.kind == "model_loaded":
            self.inspecting = False
            self.inspect_button.configure(state="normal")
            self._show_model_info(event.payload)
            self._log("info", f"Checkpoint cargado: {event.payload['checkpoint_name']}")
        elif event.kind == "model_failed":
            self.inspecting = False
            self.inspect_button.configure(state="normal")
            self.model_info = None
            self.model_status_var.set("No se pudo cargar el checkpoint.")
            self._set_text(self.model_info_text, event.payload or event.message)
            self._log("error", event.message)
            messagebox.showerror("Carga del modelo", event.message[:2500])
        elif event.kind == "status":
            self.status_var.set(event.message)
        elif event.kind == "warning":
            self._log("warning", event.message)
        elif event.kind == "log":
            self._log("info", event.message)
        elif event.kind == "progress":
            self.progress.configure(
                maximum=max(1, event.total),
                value=min(event.completed, max(1, event.total)),
            )
            self.status_var.set(
                f"{event.completed}/{event.total} · {event.message}"
            )
        elif event.kind == "publishing":
            self.status_var.set(event.message)
            self.cancel_button.configure(state="disabled")
        elif event.kind == "task_completed":
            self.processing = False
            self.start_button.configure(state="normal")
            self.cancel_button.configure(state="disabled")
            for result in event.payload:
                self.results[str(result.source)] = result
                self.results_table.insert(
                    "",
                    "end",
                    iid=str(result.source),
                    text=result.source.name,
                    values=(
                        f"{result.metadata.sample_rate:,}",
                        result.metadata.channels,
                        f"{result.metadata.duration_seconds:.2f} s",
                        str(result.output),
                    ),
                )
            self.status_var.set(f"Terminado: {len(event.payload)} archivo(s)")
            self._log("info", f"Procesamiento terminado: {len(event.payload)} salida(s)")
            self._refresh_history()
        elif event.kind == "task_cancelled":
            self.processing = False
            self.start_button.configure(state="normal")
            self.cancel_button.configure(state="disabled")
            self.status_var.set(event.message)
            self._log("warning", event.message)
            self._refresh_history()
        elif event.kind == "task_failed":
            self.processing = False
            self.start_button.configure(state="normal")
            self.cancel_button.configure(state="disabled")
            self.status_var.set("Error en el procesamiento")
            self._log("error", event.message)
            messagebox.showerror("Error de inferencia", event.message[:3000])
            self._refresh_history()

        if self.close_requested and not self.processing and not self.inspecting:
            self.root.destroy()

    def _drain_events(self) -> None:
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            self._handle_event(event)
        if self.root.winfo_exists():
            self.root.after(100, self._drain_events)

    def _refresh_history(self) -> None:
        if not hasattr(self, "history_table"):
            return
        self.history_table.delete(*self.history_table.get_children())
        try:
            records = read_history()
        except (OSError, ValueError) as error:
            self._log("error", str(error))
            return
        for index, record in enumerate(reversed(records)):
            self.history_table.insert(
                "",
                "end",
                iid=f"history-{index}",
                text=record.get("finished_at", record.get("started_at", "")),
                values=(
                    record.get("status", "desconocido"),
                    len(record.get("files", [])),
                    Path(record.get("checkpoint", "")).name,
                    record.get("output_directory", ""),
                ),
            )

    def _play_selected(self, which: str) -> None:
        selection = self.results_table.selection()
        if not selection:
            messagebox.showinfo("Reproducción", "Selecciona un resultado.")
            return
        result = self.results.get(selection[0])
        if result is None:
            return
        path = result.source if which == "input" else result.output
        if path.suffix.lower() != ".wav":
            messagebox.showinfo(
                "Reproducción",
                "La reproducción integrada admite WAV sin modificar el archivo. "
                "FLAC sigue disponible para el pipeline y puede abrirse con otra app.",
            )
            return
        if sys.platform != "win32":
            messagebox.showerror(
                "Reproducción", "La reproducción integrada está disponible en Windows."
            )
            return
        try:
            import winsound

            winsound.PlaySound(
                str(path), winsound.SND_FILENAME | winsound.SND_ASYNC
            )
        except RuntimeError as error:
            self._log("error", f"No se pudo reproducir {path}: {error}")
            messagebox.showerror("Reproducción", str(error))

    def _stop_playback(self) -> None:
        if sys.platform != "win32":
            return
        import winsound

        winsound.PlaySound(None, 0)

    def _log(self, level: str, message: str) -> None:
        log_method = getattr(self.logger, level)
        log_method(message)
        if not hasattr(self, "log_text"):
            return
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"[{level.upper()}] {message}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    @staticmethod
    def _set_text(widget: tk.Text, value: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", value)
        widget.configure(state="disabled")

    def _close(self) -> None:
        if self.processing:
            self.close_requested = True
            self._cancel_processing()
            return
        if self.inspecting:
            self.close_requested = True
            self.status_var.set("Esperando que finalice la inspección del checkpoint…")
            return
        self.root.destroy()


def run_app() -> None:
    root = tk.Tk()
    SGMSEApp(root)
    root.mainloop()
