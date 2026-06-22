"""
Model Health and System Logger.

Discipline: Software Engineering / MLOps
Refine means here: systematic monitoring of
every component so failures are caught, logged
and handled gracefully instead of crashing.

What this adds that was missing:
    1. Structured logging across all modules
    2. Model cache so models load once not every run
    3. Graceful degradation — if one module fails
       the rest keep running
    4. Processing time tracking per module
    5. Memory usage monitoring
    6. Pre-flight checks before processing starts
    7. Version tracking for reproducibility

Why this matters for your paper:
    Chapter 4 needs to describe system reliability.
    Examiners ask: what happens when it fails?
    This answers that question with evidence.

Author: African LLM Project
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Generator

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

formatter = logging.Formatter(
    fmt="%(asctime)s [%(levelname)-8s] "
        "%(name)-20s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

file_handler = logging.FileHandler(
    LOG_DIR / "system.log",
    encoding="utf-8",
)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.INFO)

root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

log = logging.getLogger("model_health")



@dataclass
class ModuleStatus:
    """Tracks the health of one pipeline module."""

    name: str
    status: str = "not_run"  
    start_time: float = 0.0
    end_time: float = 0.0
    error_message: str = ""
    output_files: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        if self.end_time and self.start_time:
            return round(self.end_time - self.start_time, 2)
        return 0.0

    @property
    def is_ok(self) -> bool:
        return self.status == "ok"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "duration_seconds": self.duration_seconds,
            "error": self.error_message,
            "output_files": self.output_files,
            "metrics": self.metrics,
        }


@dataclass
class SystemHealth:
    """Overall system health report."""

    run_id: str = ""
    start_time: str = ""
    end_time: str = ""
    python_version: str = ""
    modules: list[ModuleStatus] = field(
        default_factory=list
    )
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.run_id:
            self.run_id = datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )
        if not self.start_time:
            self.start_time = datetime.now().isoformat()
        self.python_version = sys.version.split()[0]

    @property
    def all_ok(self) -> bool:
        return all(m.is_ok for m in self.modules)

    @property
    def failed_modules(self) -> list[ModuleStatus]:
        return [
            m for m in self.modules
            if m.status == "error"
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "python_version": self.python_version,
            "overall_status": (
                "ok" if self.all_ok else "degraded"
            ),
            "modules": [m.to_dict() for m in self.modules],
            "warnings": self.warnings,
            "failed_count": len(self.failed_modules),
        }



_MODEL_CACHE: dict[str, Any] = {}


def get_cached_model(
    model_name: str,
    loader_fn: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """
    Returns model from cache or loads it fresh.

    Args:
        model_name: Unique cache key
        loader_fn: Function that loads the model
        *args: Arguments for loader_fn
        **kwargs: Keyword arguments for loader_fn

    Returns:
        Loaded model object
    """
    if model_name not in _MODEL_CACHE:
        log.info("Loading model: %s", model_name)
        start = time.time()
        _MODEL_CACHE[model_name] = loader_fn(
            *args, **kwargs
        )
        elapsed = time.time() - start
        log.info(
            "Model loaded in %.2fs: %s",
            elapsed, model_name,
        )
    else:
        log.debug("Cache hit: %s", model_name)

    return _MODEL_CACHE[model_name]


def clear_model_cache() -> None:
    """Clears all cached models to free memory."""
    count = len(_MODEL_CACHE)
    _MODEL_CACHE.clear()
    log.info("Cleared %d models from cache", count)



@contextmanager
def module_runner(
    health: SystemHealth,
    module_name: str,
    required_files: list[str] | None = None,
) -> Generator[ModuleStatus, None, None]:
    """
    Context manager for safe module execution.

    Handles:
    - Timing
    - Error catching with full traceback to log
    - Pre-flight file checks
    - Status updating

    Usage:
        with module_runner(health, "audio") as status:
            result = transcribe_audio("audio.wav")
            status.metrics["words"] = len(result.split())

    Args:
        health: SystemHealth object to update
        module_name: Human-readable module name
        required_files: Files that must exist before running

    Yields:
        ModuleStatus object for the caller to update
    """
    status = ModuleStatus(name=module_name)
    health.modules.append(status)

    if required_files:
        missing = [
            f for f in required_files
            if not Path(f).exists()
        ]
        if missing:
            status.status = "error"
            status.error_message = (
                f"Required files missing: "
                f"{', '.join(missing)}"
            )
            log.error(
                "[%s] Missing required files: %s",
                module_name, missing,
            )
            yield status
            return

    status.start_time = time.time()
    log.info("[%s] Starting", module_name)

    try:
        yield status
        status.status = "ok"
        log.info(
            "[%s] Completed in %.2fs",
            module_name, status.duration_seconds,
        )

    except FileNotFoundError as e:
        status.status = "error"
        status.error_message = f"File not found: {e}"
        log.error(
            "[%s] File not found: %s",
            module_name, e,
        )

    except MemoryError:
        status.status = "error"
        status.error_message = (
            "Out of memory. Try a shorter video "
            "or close other applications."
        )
        log.error("[%s] Out of memory", module_name)

    except Exception as e:
        status.status = "error"
        status.error_message = str(e)
        log.debug(
            "[%s] Full traceback:\n%s",
            module_name, traceback.format_exc(),
        )
        log.error(
            "[%s] Failed: %s",
            module_name, type(e).__name__,
        )

    finally:
        status.end_time = time.time()



def check_dependencies() -> dict[str, bool]:
    """
    Checks all required Python packages are installed.

    Returns:
        Dict mapping package name to availability
    """
    required_packages = {
        "torch":        "PyTorch (deep learning)",
        "transformers": "HuggingFace Transformers",
        "cv2":          "OpenCV (video processing)",
        "whisper":      "Whisper ASR",
        "numpy":        "NumPy",
        "datasets":     "HuggingFace Datasets",
        "streamlit":    "Streamlit (web interface)",
        "sacrebleu":    "SacreBLEU (evaluation)",
    }

    results: dict[str, bool] = {}
    all_ok = True

    print("\n── Dependency Check ──────────────────────")
    for package, description in required_packages.items():
        try:
            __import__(package)
            results[package] = True
            print(f"  ✅ {package:<15} {description}")
        except ImportError:
            results[package] = False
            all_ok = False
            print(f"  ❌ {package:<15} {description}")
            log.warning("Missing package: %s", package)

    if not all_ok:
        missing = [p for p, ok in results.items() if not ok]
        print(f"\n  Install missing: pip install "
              f"{' '.join(missing)}")

    return results


def check_required_files() -> dict[str, bool]:
    """
    Checks all required files exist.

    Returns:
        Dict mapping file path to existence
    """
    required: dict[str, str] = {
        "corpus.json":
            "Ibibio corpus (run build_corpus.py)",
        "transcript.json":
            "Audio transcript (run audio_module.py)",
        "processed_text.json":
            "Processed text (run text_module.py)",
        "fused_data.json":
            "Fused data (run fusion_module.py)",
        "transformer_output.json":
            "Transformer output (run transformer_module.py)",
        "translations.json":
            "Translations (run translation_module.py)",
    }

    optional: dict[str, str] = {
        "ibibio_model/config.json":
            "Fine-tuned Ibibio model",
        "ibibio_architecture.json":
            "Linguistic architecture",
        "advanced_evaluation_results.json":
            "Evaluation results",
    }

    results: dict[str, bool] = {}

    print("\n── Required Files ────────────────────────")
    for filepath, description in required.items():
        exists = Path(filepath).exists()
        results[filepath] = exists
        icon = "✅" if exists else "❌"
        print(f"  {icon} {filepath:<35} {description}")

    print("\n── Optional Files ────────────────────────")
    for filepath, description in optional.items():
        exists = Path(filepath).exists()
        results[filepath] = exists
        icon = "✅" if exists else "⚠️ "
        print(f"  {icon} {filepath:<35} {description}")

    return results


def check_system_resources() -> dict[str, Any]:
    """
    Checks available system resources.

    Returns:
        Dict with memory and disk info
    """
    import shutil

    resources: dict[str, Any] = {}

    # Disk space
    disk = shutil.disk_usage(".")
    free_gb = disk.free / (1024 ** 3)
    resources["disk_free_gb"] = round(free_gb, 2)

    # Python memory (basic)
    try:
        import psutil
        mem = psutil.virtual_memory()
        resources["ram_total_gb"] = round(
            mem.total / (1024 ** 3), 1
        )
        resources["ram_available_gb"] = round(
            mem.available / (1024 ** 3), 1
        )
        resources["ram_percent_used"] = mem.percent
    except ImportError:
        resources["ram_note"] = (
            "Install psutil for memory monitoring"
        )

    print("\n── System Resources ──────────────────────")
    print(f"  Disk free:      {free_gb:.1f} GB")
    if "ram_total_gb" in resources:
        print(
            f"  RAM total:      "
            f"{resources['ram_total_gb']} GB"
        )
        print(
            f"  RAM available:  "
            f"{resources['ram_available_gb']} GB"
        )
        print(
            f"  RAM used:       "
            f"{resources['ram_percent_used']}%"
        )

        if resources["ram_available_gb"] < 2.0:
            log.warning(
                "Low available RAM: %.1f GB",
                resources["ram_available_gb"],
            )
            print(
                "  ⚠️  Low RAM — "
                "close other applications"
            )

    if free_gb < 5.0:
        log.warning(
            "Low disk space: %.1f GB free", free_gb
        )
        print("  ⚠️  Low disk space")

    return resources


SYSTEM_VERSION = {
    "system": "African Multilingual Video LLM",
    "version": "1.0.0",
    "corpus_version": "1.0",
    "model_version": "nllb-200-distilled-600M",
    "ibibio_model_version": "finetuned-v1",
    "python_min": "3.8",
    "created": "2026",
    "languages_supported": [
        "English (input)",
        "Yoruba",
        "Hausa",
        "Igbo",
        "Ibibio (fine-tuned)",
    ],
}


def save_run_report(health: SystemHealth) -> Path:
    """
    Saves a full run report to the logs folder.

    Args:
        health: Completed SystemHealth object

    Returns:
        Path to saved report
    """
    health.end_time = datetime.now().isoformat()
    report = {
        **health.to_dict(),
        "system_version": SYSTEM_VERSION,
    }

    report_path = (
        LOG_DIR / f"run_{health.run_id}.json"
    )
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    log.info("Run report saved: %s", report_path)
    return report_path



def run_system_check() -> SystemHealth:
    """
    Runs a complete system health check.
    Call this before starting any processing.

    Returns:
        SystemHealth object with all results
    """
    print("=" * 50)
    print("SYSTEM HEALTH CHECK")
    print(f"Version: {SYSTEM_VERSION['version']}")
    print("=" * 50)

    health = SystemHealth()

    deps = check_dependencies()
    missing_critical = [
        p for p, ok in deps.items()
        if not ok and p in ["torch", "transformers"]
    ]
    if missing_critical:
        health.warnings.append(
            f"Critical packages missing: "
            f"{missing_critical}"
        )

    files = check_required_files()
    missing_files = [
        f for f, exists in files.items() if not exists
    ]
    if missing_files:
        health.warnings.append(
            f"Missing files: {len(missing_files)}"
        )

    resources = check_system_resources()

    print("\n── Summary ───────────────────────────────")
    warn_count = len(health.warnings)
    if warn_count == 0:
        print("  ✅ All checks passed")
        print("  System ready for processing")
    else:
        print(f"  ⚠️  {warn_count} warnings found")
        for w in health.warnings:
            print(f"     • {w}")

    print("=" * 50)

    save_run_report(health)

    return health


if __name__ == "__main__":
    health = run_system_check()
