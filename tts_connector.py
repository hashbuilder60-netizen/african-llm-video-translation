"""
TTS Connector and Output Bundler.

Discipline: Speech Synthesis / Software Engineering
Refine means here: connecting every output the system
produces into one unified audio + subtitle + download
package that the user can take anywhere.

What this does:
    1. Generates TTS audio for every selected language
       - Ibibio: phoneme-based engine (offline, original)
       - Other languages: gTTS (needs internet)
       - Offline fallback: pyttsx3 if gTTS unavailable

    2. Bundles all outputs into a ZIP file:
       - All SRT subtitle files (standard + bilingual)
       - All TTS audio files per language
       - Transcript (txt + json)
       - Evaluation results
       - Corpus sample
       - Chapter 4 and 5 text
       - Manifest listing everything

Author: African LLM Project
"""

from __future__ import annotations

import io
import json
import logging
import os
import zipfile
from pathlib import Path
from typing import Any

log = logging.getLogger("tts_connector")


# ── TTS for all languages ────────────────────────

def generate_ibibio_audio(
    text: str,
    output_path: str,
) -> str | None:
    """
    Generates Ibibio speech.

    Priority order:
    1. Real Ibibio voice from AfricanVoices/LLSTI
       recordings — actual native speaker audio
    2. Phoneme-based engine — synthetic but original
    3. gTTS with Yoruba proxy — intelligible fallback

    Args:
        text: Ibibio text to synthesise
        output_path: WAV output path

    Returns:
        Path to saved file or None
    """
    # Strategy 1 — Real Ibibio voice recordings
    try:
        from ibibio_voice import speak_ibibio
        audio_path, method = speak_ibibio(
            text,
            output_path=output_path,
            data_dir="africanvoices_data",
        )
        if audio_path and method in (
            "real_voice", "concatenated"
        ):
            log.info(
                "Real Ibibio voice used (%s)", method
            )
            return audio_path
    except ImportError:
        log.debug("ibibio_voice.py not found")
    except Exception as e:
        log.warning("Real voice failed: %s", e)

    # Strategy 2 — Phoneme engine
    try:
        #from ibibio_tts import synthesize_text
        result = synthesize_text(
            text, output_path, speaking_rate=0.85
        )
        if result:
            log.info("Phoneme TTS used")
            return result
    except Exception as e:
        log.warning("Phoneme TTS failed: %s", e)

    # Strategy 3 — gTTS Yoruba proxy
    try:
        mp3_path = output_path.replace(".wav", ".mp3")
        result = generate_gtts_audio(
            text, mp3_path, lang_code="yo"
        )
        if result:
            log.info("gTTS Yoruba proxy used")
            return result
    except Exception as e:
        log.warning("gTTS fallback failed: %s", e)

    return None


def generate_gtts_audio(
    text: str,
    output_path: str,
    lang_code: str = "en",
) -> str | None:
    """
    Generates audio using gTTS for languages
    supported by Google TTS.
    Requires internet connection.

    Args:
        text: Text to synthesise
        output_path: MP3 output path
        lang_code: gTTS language code

    Returns:
        Path to saved file or None
    """
    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang=lang_code, slow=False)
        tts.save(output_path)
        log.info("gTTS saved: %s", output_path)
        return output_path
    except Exception as e:
        log.warning("gTTS failed (%s): %s", lang_code, e)
        return None


def generate_pyttsx3_audio(
    text: str,
    output_path: str,
) -> str | None:
    """
    Offline fallback TTS using pyttsx3.
    Works without internet but English only.

    Args:
        text: Text to synthesise
        output_path: Output path

    Returns:
        Path to saved file or None
    """
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
        engine.setProperty("volume", 0.9)
        engine.save_to_file(text, output_path)
        engine.runAndWait()
        return output_path
    except Exception as e:
        log.warning("pyttsx3 failed: %s", e)
        return None


def generate_language_audio(
    lang_key: str,
    lang_info: dict[str, Any],
    text: str,
    output_dir: Path,
) -> str | None:
    """
    Generates TTS audio for one language.
    Chooses the right engine based on language config.

    Priority:
    1. Ibibio: phoneme engine (offline)
    2. gTTS supported: gTTS (online)
    3. Fallback: pyttsx3 (offline, English only)

    Args:
        lang_key: Language key
        lang_info: Language config from registry
        text: Text to synthesise
        output_dir: Output directory

    Returns:
        Path to audio file or None
    """
    if not text or not text.strip():
        return None

    model_type = lang_info.get("model_type", "nllb200")
    tts_type = lang_info.get("tts_type")
    gtts_code = lang_info.get("gtts_code")

    if model_type == "finetuned" and tts_type == "phoneme":
        out_path = str(output_dir / f"{lang_key}_tts.wav")
        result = generate_ibibio_audio(text, out_path)
        if result:
            log.info("Ibibio phoneme TTS: %s", out_path)
            return result

    if tts_type == "gtts" and gtts_code:
        out_path = str(output_dir / f"{lang_key}_tts.mp3")
        result = generate_gtts_audio(text, out_path, gtts_code)
        if result:
            return result

    out_path = str(output_dir / f"{lang_key}_tts_en.mp3")
    result = generate_pyttsx3_audio(
        f"Translation in {lang_info.get('name', lang_key)}. "
        f"{text[:200]}",
        out_path,
    )
    return result


def generate_all_tts(
    selected_languages: list[str],
    registry: dict[str, Any],
    output_dir: str = "tts_output",
) -> dict[str, str]:
    """
    Generates TTS audio for all selected languages.

    For each language:
    - Loads the translation from segment JSON
    - Takes first 5 segments as representative sample
    - Generates audio
    - Saves to tts_output/{lang_key}_tts.{wav|mp3}

    Args:
        selected_languages: Language keys
        registry: Language registry
        output_dir: Output directory

    Returns:
        Dict mapping lang_key to audio file path
    """
    print("=" * 50)
    print("TTS GENERATION — ALL LANGUAGES")
    print("=" * 50)

    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True)

    audio_files: dict[str, str] = {}
    seg_dir = Path("segment_translations")

    for lang_key in selected_languages:
        lang_info = registry.get("languages", {}).get(
            lang_key, {}
        )
        lang_name = lang_info.get("name", lang_key)

        print(f"\n  Generating audio: {lang_name}...")

        seg_file = seg_dir / f"{lang_key}_segments.json"
        text = ""

        if seg_file.exists():
            with seg_file.open(encoding="utf-8") as f:
                segs = json.load(f)
            sample_segs = segs[:5]
            text = " ".join(
                s.get("translation", "")
                for s in sample_segs
                if s.get("translation", "").strip()
            )

        if not text:
            if lang_key == "ibibio":
                ibb_path = Path("ibibio_test_results.json")
                if ibb_path.exists():
                    with ibb_path.open(
                        encoding="utf-8"
                    ) as f:
                        ibb_data = json.load(f)
                    text = " ".join(
                        r.get("ibibio_translation", "")
                        for r in ibb_data[:5]
                    )

        if not text:
            print(f"    No text found for {lang_name}")
            continue

        result = generate_language_audio(
            lang_key, lang_info, text, out_dir
        )

        if result:
            audio_files[lang_key] = result
            print(f"    ✓ {lang_name}: {result}")
        else:
            print(f"    ✗ {lang_name}: audio generation failed")

    demo_phrases = [
        ("Ememe nnyin", "greeting"),
        ("Sosongo", "thank_you"),
        ("Ami okut fo", "love"),
        ("Edisua edi mme ekpuk", "education_important"),
        ("Nyin edi nte", "we_are_one"),
        ("Abasi mbot fo", "god_bless"),
        ("Ndito edi iman", "health_wealth"),
        ("Eket nyin edi mme", "language_beautiful"),
    ]

    demo_dir = out_dir / "demo_phrases"
    demo_dir.mkdir(exist_ok=True)

    print("\n  Generating Ibibio phrase demos...")
    for phrase, name in demo_phrases:
        out_path = str(demo_dir / f"{name}.wav")
        if not Path(out_path).exists():
            result = generate_ibibio_audio(phrase, out_path)
            if result:
                print(f"    ✓ {phrase}")

    print("\n" + "=" * 50)
    print(f"TTS COMPLETE — {len(audio_files)} languages")
    print("=" * 50)

    return audio_files


def create_download_bundle(
    selected_languages: list[str] | None = None,
    include_audio: bool = True,
    include_subtitles: bool = True,
    include_corpus: bool = True,
    include_chapters: bool = True,
    include_evaluation: bool = True,
) -> bytes:
    """
    Creates a ZIP archive of all project outputs.

    The user downloads this ZIP and extracts it
    wherever they want on their machine.

    Contents:
        subtitles/          SRT files per language
        audio/              TTS WAV/MP3 per language
        audio/demo_phrases/ Ibibio phrase audio
        data/               Transcript, translations JSON
        evaluation/         BLEU, chrF, METEOR results
        corpus/             Ibibio corpus JSON
        chapters/           Chapter 4 and 5 text
        manifest.txt        Full list of contents

    Args:
        selected_languages: Languages to include
        include_audio: Whether to include audio files
        include_subtitles: Whether to include SRT files
        include_corpus: Whether to include corpus
        include_chapters: Whether to include chapters
        include_evaluation: Whether to include eval results

    Returns:
        ZIP file as bytes for st.download_button
    """
    buffer = io.BytesIO()

    with zipfile.ZipFile(
        buffer, "w", zipfile.ZIP_DEFLATED
    ) as zf:
        manifest_lines = [
            "AFRICAN MULTILINGUAL VIDEO LLM",
            "Output Bundle",
            "Kwara State University, Malete",
            "Department of Computer Science — Group 14",
            "=" * 50,
            "",
        ]

        if include_subtitles:
            seg_dir = Path("segment_translations")
            if seg_dir.exists():
                srt_count = 0
                manifest_lines.append("SUBTITLE FILES:")

                langs_to_include = selected_languages
                if not langs_to_include:
                    langs_to_include = [
                        p.stem.replace("_segments", "")
                        for p in seg_dir.glob("*_segments.json")
                    ]

                for lang_key in (langs_to_include or []):
                    srt = seg_dir / f"{lang_key}.srt"
                    bi  = seg_dir / f"{lang_key}_bilingual.srt"
                    seg = seg_dir / f"{lang_key}_segments.json"

                    if srt.exists():
                        zf.write(
                            srt,
                            f"subtitles/{lang_key}.srt",
                        )
                        manifest_lines.append(
                            f"  subtitles/{lang_key}.srt"
                        )
                        srt_count += 1

                    if bi.exists():
                        zf.write(
                            bi,
                            f"subtitles/{lang_key}_bilingual.srt",
                        )
                        manifest_lines.append(
                            f"  subtitles/{lang_key}_bilingual.srt"
                        )

                    if seg.exists():
                        zf.write(
                            seg,
                            f"subtitles/{lang_key}_data.json",
                        )

                manifest_lines.append(
                    f"  Total: {srt_count} subtitle files"
                )
                manifest_lines.append("")

        if include_audio:
            tts_dir = Path("tts_output")
            if tts_dir.exists():
                audio_count = 0
                manifest_lines.append("AUDIO FILES:")

                for audio_file in tts_dir.glob("*.wav"):
                    zf.write(
                        audio_file,
                        f"audio/{audio_file.name}",
                    )
                    manifest_lines.append(
                        f"  audio/{audio_file.name}"
                    )
                    audio_count += 1

                for audio_file in tts_dir.glob("*.mp3"):
                    zf.write(
                        audio_file,
                        f"audio/{audio_file.name}",
                    )
                    manifest_lines.append(
                        f"  audio/{audio_file.name}"
                    )
                    audio_count += 1

                demo_dir = tts_dir / "demo_phrases"
                if demo_dir.exists():
                    for wav in demo_dir.glob("*.wav"):
                        zf.write(
                            wav,
                            f"audio/demo_phrases/{wav.name}",
                        )
                        manifest_lines.append(
                            f"  audio/demo_phrases/{wav.name}"
                        )
                        audio_count += 1

                manifest_lines.append(
                    f"  Total: {audio_count} audio files"
                )
                manifest_lines.append("")

        for fname in [
            "transcript.txt",
            "transcript.json",
            "translations.json",
            "transformer_output.json",
        ]:
            p = Path(fname)
            if p.exists():
                zf.write(p, f"data/{fname}")

        manifest_lines.append("DATA FILES:")
        manifest_lines.append("  data/transcript.txt")
        manifest_lines.append("  data/transcript.json")
        manifest_lines.append("  data/translations.json")
        manifest_lines.append("")

        if include_evaluation:
            manifest_lines.append("EVALUATION:")
            for fname in [
                "evaluation_results.json",
                "advanced_evaluation_results.json",
                "ibibio_test_results.json",
                "augmentation_report.json",
            ]:
                p = Path(fname)
                if p.exists():
                    zf.write(p, f"evaluation/{fname}")
                    manifest_lines.append(
                        f"  evaluation/{fname}"
                    )
            manifest_lines.append("")

        if include_corpus:
            manifest_lines.append("CORPUS:")
            for fname in [
                "corpus.json",
                "corpus.en",
                "corpus.ibb",
                "ibibio_architecture.json",
                "ibibio_linguistics.json",
            ]:
                p = Path(fname)
                if p.exists():
                    zf.write(p, f"corpus/{fname}")
                    manifest_lines.append(
                        f"  corpus/{fname}"
                    )
            manifest_lines.append("")

        if include_chapters:
            manifest_lines.append("RESEARCH CHAPTERS:")
            for fname in [
                "chapter_4_implementation.txt",
                "chapter_5_results.txt",
            ]:
                p = Path(fname)
                if p.exists():
                    zf.write(p, f"chapters/{fname}")
                    manifest_lines.append(
                        f"  chapters/{fname}"
                    )
                else:
                    try:
                        from chapter_writer import (
                            write_all_chapters,
                        )
                        write_all_chapters()
                        if p.exists():
                            zf.write(
                                p, f"chapters/{fname}"
                            )
                            manifest_lines.append(
                                f"  chapters/{fname} "
                                f"(auto-generated)"
                            )
                    except Exception:
                        pass
            manifest_lines.append("")

        p = Path("languages.json")
        if p.exists():
            zf.write(p, "config/languages.json")
            manifest_lines.append(
                "CONFIG: config/languages.json"
            )

        instructions = """AFRICAN MULTILINGUAL VIDEO LLM — OUTPUT BUNDLE
Kwara State University, Malete
Department of Computer Science — Group 14
================================================

HOW TO USE THESE FILES
======================

SUBTITLE FILES (subtitles/)
----------------------------
Load .srt files in VLC Media Player:
1. Open your video in VLC
2. Go to Subtitle menu → Add Subtitle File
3. Select the .srt file for your chosen language
4. Subtitles will appear synced to the video

Standard (.srt)   — Translation only
Bilingual (.srt)  — English + Translation together

AUDIO FILES (audio/)
--------------------
- *_tts.wav files: Ibibio phoneme-synthesised audio
- *_tts.mp3 files: gTTS audio for other languages
- demo_phrases/: Individual Ibibio phrase audio

Play in VLC, Windows Media Player, or any audio player.

DATA FILES (data/)
------------------
- transcript.txt: Full spoken text from the video
- transcript.json: Timestamped segments from Whisper
- translations.json: Summary translations per language

EVALUATION (evaluation/)
------------------------
- evaluation_results.json: BLEU scores
- advanced_evaluation_results.json: chrF + METEOR + WER

CORPUS (corpus/)
----------------
- corpus.json: Full 438-pair Ibibio-English corpus
- corpus.en / corpus.ibb: Parallel text files
- ibibio_architecture.json: Phonological analysis

CHAPTERS (chapters/)
--------------------
- chapter_4_implementation.txt: Implementation chapter
- chapter_5_results.txt: Results and evaluation chapter
Copy these into your Word document and format accordingly.

================================================
Research Prototype v3.0 — 2026
"""
        zf.writestr(
            "HOW_TO_USE.txt", instructions
        )

        manifest_text = "\n".join(manifest_lines)
        zf.writestr("manifest.txt", manifest_text)

    buffer.seek(0)
    return buffer.getvalue()


def get_bundle_size_estimate() -> str:
    """
    Estimates the download bundle size.

    Returns:
        Human-readable size string
    """
    total_bytes = 0

    paths_to_check = [
        "segment_translations/",
        "tts_output/",
        "corpus.json",
        "evaluation_results.json",
        "transcript.json",
        "translations.json",
        "chapter_4_implementation.txt",
        "chapter_5_results.txt",
    ]

    for path_str in paths_to_check:
        p = Path(path_str)
        if p.is_dir():
            total_bytes += sum(
                f.stat().st_size
                for f in p.rglob("*")
                if f.is_file()
            )
        elif p.exists():
            total_bytes += p.stat().st_size

    estimated = int(total_bytes * 0.5)

    if estimated < 1024 * 1024:
        return f"{estimated // 1024} KB"
    else:
        return f"{estimated // (1024 * 1024):.1f} MB"
