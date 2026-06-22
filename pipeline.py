"""
Master Pipeline.

Discipline: Software Engineering / MLOps
Refine means here: orchestrating all modules
in the correct order with proper error handling,
logging and graceful degradation.

What this replaces:
    Running each module manually one by one.
    Now one command runs everything in sequence,
    passes data between modules automatically,
    and produces a final report.

Author: African LLM Project
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

from model_health import (
    SystemHealth,
    module_runner,
    run_system_check,
    save_run_report,
)

log = logging.getLogger("pipeline")


def run_pipeline(
    video_path: str,
    run_health_check: bool = True,
) -> dict[str, Any]:
    """
    Runs the complete pipeline from video to translations.

    Modules run in order:
        1. Video processing
        2. Audio transcription
        3. Text processing
        4. Multimodal fusion
        5. Transformer backbone
        6. Translation output

    Args:
        video_path: Path to input video file
        run_health_check: Whether to check system first

    Returns:
        Dict with all pipeline results
    """
    print("=" * 55)
    print("AFRICAN MULTILINGUAL VIDEO LLM PIPELINE")
    print("=" * 55)

    if run_health_check:
        health = run_system_check()
    else:
        health = SystemHealth()

    results: dict[str, Any] = {
        "video_path": video_path,
        "status": "running",
        "modules": {},
    }

    with module_runner(
        health, "VideoProcessing"
    ) as status:
        from video_module import process_video

        frames, audio = process_video(video_path)

        if frames and audio:
            status.metrics["frames_extracted"] = len(frames)
            status.metrics["audio_path"] = audio
            status.output_files = [audio]
            results["modules"]["video"] = {
                "frames": len(frames),
                "audio": audio,
            }
            log.info(
                "Video processed: %d frames, audio: %s",
                len(frames), audio,
            )
        else:
            raise RuntimeError(
                "Video processing returned no output"
            )

    if not status.is_ok:
        results["status"] = "failed_at_video"
        save_run_report(health)
        return results

    with module_runner(
        health, "AudioTranscription",
        required_files=["audio.wav"],
    ) as status:
        from audio_module import transcribe_audio

        transcript = transcribe_audio("audio.wav")

        if transcript:
            status.metrics["word_count"] = (
                len(transcript.split())
            )
            status.output_files = [
                "transcript.txt", "transcript.json"
            ]
            results["modules"]["audio"] = {
                "word_count": len(transcript.split()),
                "transcript_preview": transcript[:200],
            }
        else:
            raise RuntimeError(
                "Transcription returned no text"
            )

    with module_runner(
        health, "TextProcessing",
        required_files=["transcript.json"],
    ) as status:
        from text_module import process_text

        text_data = process_text("transcript.json")

        if text_data:
            status.metrics["sentences"] = len(
                text_data.get("all_sentences", [])
            )
            status.output_files = ["processed_text.json"]
            results["modules"]["text"] = {
                "sentences": status.metrics["sentences"],
            }

    with module_runner(
        health, "MultimodalFusion",
        required_files=[
            "transcript.json",
            "processed_text.json",
        ],
    ) as status:
        from fusion_module import run_fusion

        fused = run_fusion()

        if fused:
            status.metrics["segments"] = fused.get(
                "total_segments", 0
            )
            status.output_files = ["fused_data.json"]
            results["modules"]["fusion"] = {
                "segments": status.metrics["segments"],
            }

    with module_runner(
        health, "TransformerBackbone",
        required_files=["fused_data.json"],
    ) as status:
        from transformer_module import (
            process_with_transformer,
        )

        transformer_output = process_with_transformer()

        if transformer_output:
            status.metrics["topics"] = len(
                transformer_output.get("key_topics", [])
            )
            status.output_files = [
                "transformer_output.json"
            ]
            results["modules"]["transformer"] = {
                "summary": transformer_output.get(
                    "video_summary", ""
                )[:300],
                "topics": transformer_output.get(
                    "key_topics", []
                ),
            }

    with module_runner(
        health, "TranslationOutput",
        required_files=["transformer_output.json"],
    ) as status:
        from translation_module import (
            load_translation_model,
            translate_to_all_languages,
        )

        with open(
            "transformer_output.json",
            encoding="utf-8",
        ) as f:
            t_output = json.load(f)

        summary = t_output.get("video_summary", "")
        short_summary = " ".join(
            summary.split()[:200]
        )

        tokenizer, model = load_translation_model()
        translations = translate_to_all_languages(
            short_summary, tokenizer, model
        )

        status.output_files = ["translations.json"]
        results["modules"]["translation"] = {
            "languages": list(translations.keys()),
            "translations": translations,
        }

    failed = [
        m.name for m in health.modules
        if m.status == "error"
    ]

    if failed:
        results["status"] = f"partial_failure: {failed}"
        log.warning(
            "Pipeline completed with failures: %s",
            failed,
        )
    else:
        results["status"] = "complete"
        log.info("Pipeline completed successfully")

    print("\n── Module Timing ─────────────────────────")
    print(f"  {'Module':<25} {'Status':<10} {'Time'}")
    print("  " + "-" * 50)
    for m in health.modules:
        icon = "✅" if m.is_ok else "❌"
        print(
            f"  {icon} {m.name:<23} "
            f"{m.status:<10} "
            f"{m.duration_seconds:.1f}s"
        )

    total_time = sum(
        m.duration_seconds for m in health.modules
    )
    print(f"\n  Total time: {total_time:.1f}s")

    report_path = save_run_report(health)
    results["run_report"] = str(report_path)

    print("\n" + "=" * 55)
    print(f"STATUS: {results['status'].upper()}")
    print("=" * 55)

    return results


if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else "test_video.mp4"
    results = run_pipeline(video)
    print(json.dumps(
        {k: v for k, v in results.items()
         if k != "modules"},
        indent=2,
    ))
