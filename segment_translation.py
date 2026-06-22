"""
Segment-Level Translation Module.

Discipline: NLP / Computational Linguistics
Refine means here: translating each individual
spoken segment with its exact timestamp, not just
summarising the whole video and translating once.

Why this matters:
    Previous approach translated only the summary.
    This approach translates every sentence spoken,
    matched to the exact second it was spoken.
    This produces real subtitles not just captions.

Architecture:
    Whisper segments (text + start + end)
        → NLLB-200 per segment
        → Translated segments with timestamps
        → SRT subtitle file per language
        → Audio per segment (TTS)

Author: African LLM Project
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger("segment_translation")


@dataclass
class TranslatedSegment:
    """One translated subtitle segment."""

    segment_id: int
    start_seconds: float
    end_seconds: float
    original_text: str
    translated_text: str
    language: str
    language_name: str

    @property
    def start_ms(self) -> float:
        return self.start_seconds * 1000

    @property
    def end_ms(self) -> float:
        return self.end_seconds * 1000

    def to_srt_cue(self) -> str:
        """Formats as SRT subtitle cue."""
        def ms_to_srt(ms: float) -> str:
            h = int(ms // 3_600_000)
            m = int((ms % 3_600_000) // 60_000)
            s = int((ms % 60_000) // 1_000)
            cs = int(ms % 1_000)
            return f"{h:02d}:{m:02d}:{s:02d},{cs:03d}"

        return (
            f"{self.segment_id}\n"
            f"{ms_to_srt(self.start_ms)} --> "
            f"{ms_to_srt(self.end_ms)}\n"
            f"{self.translated_text}\n"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "start": self.start_seconds,
            "end": self.end_seconds,
            "original": self.original_text,
            "translation": self.translated_text,
            "language": self.language,
        }


def load_language_registry(
    registry_path: str = "languages.json",
) -> dict[str, Any]:
    """
    Loads the dynamic language registry.

    Args:
        registry_path: Path to languages.json

    Returns:
        Language registry dict
    """
    path = Path(registry_path)
    if not path.exists():
        log.warning(
            "languages.json not found. "
            "Using default 3 languages."
        )
        return {
            "languages": {
                "yoruba": {
                    "name": "Yoruba",
                    "nllb_code": "yor_Latn",
                    "model_type": "nllb200",
                    "enabled": True,
                },
                "hausa": {
                    "name": "Hausa",
                    "nllb_code": "hau_Latn",
                    "model_type": "nllb200",
                    "enabled": True,
                },
                "igbo": {
                    "name": "Igbo",
                    "nllb_code": "ibo_Latn",
                    "model_type": "nllb200",
                    "enabled": True,
                },
            }
        }

    with path.open(encoding="utf-8") as f:
        return json.load(f)


def get_enabled_languages(
    registry: dict[str, Any],
) -> dict[str, Any]:
    """Returns only enabled languages."""
    return {
        k: v
        for k, v in registry["languages"].items()
        if v.get("enabled", False)
    }


def get_language_display_list(
    registry: dict[str, Any],
) -> list[tuple[str, str]]:
    """
    Returns list of (key, display_name) tuples
    for use in UI dropdowns.

    Returns:
        Sorted list of (key, display_name)
    """
    enabled = get_enabled_languages(registry)
    result = []

    for key, lang in enabled.items():
        model_badge = (
            "Fine-tuned"
            if lang.get("model_type") == "finetuned"
            else "NLLB-200"
        )
        display = (
            f"{lang['name']} "
            f"({lang.get('country', '')}) "
            f"· {model_badge}"
        )
        result.append((key, display))

    result.sort(
        key=lambda x: (
            0 if "Fine-tuned" in x[1] else 1,
            x[1],
        )
    )
    return result


def load_nllb_model() -> tuple[Any, Any]:
    """
    Loads NLLB-200 model from cache.

    Returns:
        Tuple of (tokenizer, model)
    """
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
    )

    model_name = "facebook/nllb-200-distilled-600M"
    log.info("Loading NLLB-200 from cache...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name
    )
    model.eval()
    log.info("NLLB-200 loaded")
    return tokenizer, model


def load_ibibio_model() -> tuple[Any, Any] | None:
    """
    Loads the fine-tuned Ibibio model if available.

    Returns:
        Tuple of (tokenizer, model) or None
    """
    ibibio_path = Path("ibibio_model")
    if not ibibio_path.exists():
        log.warning(
            "Ibibio model not found at ibibio_model/. "
            "Run finetune_ibibio.py first."
        )
        return None

    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
    )

    log.info("Loading Ibibio fine-tuned model...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(ibibio_path)
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        str(ibibio_path)
    )
    model.eval()
    log.info("Ibibio model loaded")
    return tokenizer, model


def translate_segment(
    text: str,
    tokenizer: Any,
    model: Any,
    target_lang_code: str,
    max_length: int = 256,
) -> str:
    """
    Translates one text segment.

    Args:
        text: English text to translate
        tokenizer: Loaded tokenizer
        model: Loaded model
        target_lang_code: NLLB language code
        max_length: Maximum output length

    Returns:
        Translated text string
    """
    import torch

    if not text.strip():
        return ""

    try:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )

        target_id = tokenizer.convert_tokens_to_ids(
            target_lang_code
        )

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=target_id,
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
            )

        translation = tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True,
        )[0]

        return translation

    except Exception as e:
        log.warning(
            "Translation failed for '%s': %s",
            text[:50], e,
        )
        return text


def translate_all_segments(
    segments: list[dict[str, Any]],
    selected_languages: list[str],
    registry: dict[str, Any],
    progress_callback: Any = None,
) -> dict[str, list[TranslatedSegment]]:
    """
    Translates all Whisper segments into all
    selected languages.

    This is the core P1 improvement — segment-level
    translation using exact Whisper timestamps
    instead of summary-only translation.

    Args:
        segments: Whisper transcript segments
        selected_languages: Language keys to translate into
        registry: Language registry from languages.json
        progress_callback: Optional callback(lang, i, total)

    Returns:
        Dict mapping language key to translated segments
    """
    if not segments:
        log.error(
            "No segments found. "
            "Re-run audio_module.py with Whisper."
        )
        return {}

    log.info(
        "Translating %d segments into %d languages",
        len(segments), len(selected_languages),
    )

    nllb_tokenizer, nllb_model = load_nllb_model()
    ibibio_models = None

    results: dict[str, list[TranslatedSegment]] = {}

    for lang_key in selected_languages:
        lang_info = registry["languages"].get(lang_key)
        if not lang_info:
            log.warning("Unknown language: %s", lang_key)
            continue

        lang_name = lang_info["name"]
        model_type = lang_info.get("model_type", "nllb200")
        translated_segments: list[TranslatedSegment] = []

        log.info("Translating to %s...", lang_name)
        print(f"\n  Translating to {lang_name}...")

        if model_type == "finetuned":

            if ibibio_models is None:
                ibibio_models = load_ibibio_model()

            if ibibio_models is None:
                print(
                    f"    Skipping {lang_name} — "
                    "model not found"
                )
                continue

            tokenizer, model = ibibio_models
            lang_code = "yor_Latn" 

        else:
        
            nllb_code = lang_info.get("nllb_code")
            if not nllb_code:
                log.warning(
                    "No NLLB code for %s", lang_key
                )
                continue
            tokenizer = nllb_tokenizer
            model = nllb_model
            lang_code = nllb_code

        for idx, seg in enumerate(segments):
            original = seg.get("text", "").strip()
            if not original:
                continue

            translation = translate_segment(
                original,
                tokenizer,
                model,
                lang_code,
            )

            translated_segments.append(
                TranslatedSegment(
                    segment_id=idx + 1,
                    start_seconds=seg.get("start", 0),
                    end_seconds=seg.get("end", 0),
                    original_text=original,
                    translated_text=translation,
                    language=lang_key,
                    language_name=lang_name,
                )
            )

            if progress_callback:
                progress_callback(
                    lang_name, idx + 1, len(segments)
                )

            if (idx + 1) % 5 == 0:
                print(
                    f"    Segment {idx+1}/"
                    f"{len(segments)}: "
                    f"{original[:40]}... "
                    f"→ {translation[:40]}..."
                )

        results[lang_key] = translated_segments
        log.info(
            "Completed %s: %d segments",
            lang_name, len(translated_segments),
        )

    return results



def export_srt(
    segments: list[TranslatedSegment],
    output_path: str,
) -> str:
    """
    Exports translated segments as SRT file.

    Args:
        segments: Translated segments
        output_path: Output file path

    Returns:
        Path to saved file
    """
    Path(output_path).parent.mkdir(
        parents=True, exist_ok=True
    )

    with open(output_path, "w",
               encoding="utf-8") as f:
        for seg in segments:
            f.write(seg.to_srt_cue() + "\n")

    log.info("SRT saved: %s", output_path)
    return output_path


def export_bilingual_srt(
    segments: list[TranslatedSegment],
    output_path: str,
) -> str:
    """
    Exports SRT with both original English
    and translation on separate lines.
    This is the professional subtitle format
    used by international broadcasters.

    Args:
        segments: Translated segments
        output_path: Output file path

    Returns:
        Path to saved file
    """
    Path(output_path).parent.mkdir(
        parents=True, exist_ok=True
    )

    def ms_to_srt(ms: float) -> str:
        h = int(ms // 3_600_000)
        m = int((ms % 3_600_000) // 60_000)
        s = int((ms % 60_000) // 1_000)
        cs = int(ms % 1_000)
        return f"{h:02d}:{m:02d}:{s:02d},{cs:03d}"

    with open(output_path, "w",
               encoding="utf-8") as f:
        for seg in segments:
            f.write(f"{seg.segment_id}\n")
            f.write(
                f"{ms_to_srt(seg.start_ms)} --> "
                f"{ms_to_srt(seg.end_ms)}\n"
            )
            f.write(f"{seg.original_text}\n")
            f.write(f"{seg.translated_text}\n\n")

    log.info("Bilingual SRT saved: %s", output_path)
    return output_path


def save_all_results(
    translated: dict[str, list[TranslatedSegment]],
    output_dir: str = "segment_translations",
) -> dict[str, Any]:
    """
    Saves all translation results — JSON,
    SRT and bilingual SRT for each language.

    Args:
        translated: Dict of lang -> segments
        output_dir: Output directory

    Returns:
        Dict of saved file paths
    """
    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    saved: dict[str, Any] = {}

    for lang_key, segments in translated.items():
        if not segments:
            continue

        lang_name = segments[0].language_name

        json_path = out / f"{lang_key}_segments.json"
        with json_path.open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(
                [s.to_dict() for s in segments],
                f, indent=2, ensure_ascii=False,
            )

        srt_path = out / f"{lang_key}.srt"
        export_srt(segments, str(srt_path))

        bi_path = out / f"{lang_key}_bilingual.srt"
        export_bilingual_srt(segments, str(bi_path))

        saved[lang_key] = {
            "language_name": lang_name,
            "segment_count": len(segments),
            "json": str(json_path),
            "srt": str(srt_path),
            "bilingual_srt": str(bi_path),
        }

        print(
            f"  Saved {lang_name}: "
            f"{len(segments)} segments"
        )

    manifest_path = out / "manifest.json"
    with manifest_path.open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(saved, f, indent=2)

    log.info(
        "All results saved to %s", output_dir
    )
    return saved


def run_segment_translation(
    selected_languages: list[str] | None = None,
    transcript_path: str = "transcript.json",
) -> dict[str, Any]:
    """
    Main segment translation pipeline.

    Args:
        selected_languages: Language keys to translate.
                          None = all enabled languages.
        transcript_path: Path to Whisper transcript.

    Returns:
        Saved file paths dict
    """
    print("=" * 55)
    print("SEGMENT-LEVEL TRANSLATION")
    print("Using Whisper timestamps for accuracy")
    print("=" * 55)

    t_path = Path(transcript_path)
    if not t_path.exists():
        print(
            f"Error: {transcript_path} not found.\n"
            "Run audio_module.py first."
        )
        return {}

    with t_path.open(encoding="utf-8") as f:
        transcript = json.load(f)

    segments = transcript.get("segments", [])

    if not segments:
        print(
            "No timestamped segments found.\n"
            "Re-run audio_module.py — "
            "Whisper produces segments automatically."
        )
        return {}

    print(f"\nLoaded {len(segments)} segments")
    print(
        f"Duration: "
        f"{transcript.get('duration_seconds', 0):.1f}s"
    )

    registry = load_language_registry()
    enabled = get_enabled_languages(registry)

    if selected_languages is None:
        selected = [
            k for k in enabled
            if k != "ibibio" 
        ]
    else:
        selected = selected_languages

    print(f"\nTranslating into: {', '.join(selected)}")

    translated = translate_all_segments(
        segments, selected, registry
    )

    print("\nSaving results...")
    saved = save_all_results(translated)

    print("\n" + "=" * 55)
    print("SEGMENT TRANSLATION COMPLETE")
    print("=" * 55)
    for lang_key, info in saved.items():
        print(
            f"  {info['language_name']}: "
            f"{info['segment_count']} segments"
        )
        print(f"    SRT: {info['srt']}")
        print(f"    Bilingual: {info['bilingual_srt']}")

    print(
        "\nLoad .srt files in VLC with your video"
        "\nfor synchronised translated subtitles."
    )
    print("=" * 55)

    return saved


if __name__ == "__main__":
    run_segment_translation()
