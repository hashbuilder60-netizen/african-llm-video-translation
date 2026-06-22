"""
Ibibio Real Voice TTS Module.

Uses actual recorded Ibibio speech from the
AfricanVoices / LLSTI project instead of
synthetic phoneme generation.

How it works:
    1. Loads all Ibibio recordings + transcriptions
    2. For any input text finds the closest
       matching recorded phrase using character
       n-gram similarity
    3. Returns the actual WAV file of a real
       Ibibio speaker saying the closest phrase
    4. If confidence too low falls back to gTTS

Why this is better than our phoneme engine:
    Real human voice vs sine wave beeps.
    The LLSTI recordings are from a native
    Ibibio speaker recorded in studio conditions.

Dataset source:
    AfricanVoices (CMU neulab) / LLSTI project
    https://github.com/neulab/AfricanVoices
    Place downloaded data in: africanvoices_data/

Author: African LLM Project
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

log = logging.getLogger("ibibio_voice")



def find_ibibio_data(
    base_dir: str = "africanvoices_data",
) -> Path | None:
    """
    Finds the Ibibio audio data directory.
    Searches common locations where the
    AfricanVoices dataset might be saved.

    Returns:
        Path to Ibibio data folder or None
    """
    search_paths = [
        Path(base_dir) / "ibibio",
        Path(base_dir) / "ibb",
        Path(base_dir) / "Ibibio",
        Path(base_dir),
        Path("africanvoices_data"),
        Path("ibibio_voice_data"),
        Path("llsti_ibibio"),
    ]

    for path in search_paths:
        if not path.exists():
            continue
        wav_files = list(path.rglob("*.wav"))
        if wav_files:
            log.info(
                "Found Ibibio data at %s (%d files)",
                path, len(wav_files),
            )
            return path

    return None


def load_ibibio_recordings(
    data_dir: Path,
) -> list[dict[str, Any]]:
    """
    Loads all Ibibio recordings with their
    transcriptions.

    Searches for:
    - metadata.csv or metadata.json
    - wav/ folder with audio files
    - text/ folder with transcriptions

    Returns:
        List of dicts with 'text' and 'audio_path'
    """
    recordings: list[dict[str, Any]] = []

    csv_path = data_dir / "metadata.csv"
    if csv_path.exists():
        import csv
        with csv_path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = (
                    row.get("transcript", "")
                    or row.get("transcription", "")
                    or row.get("text", "")
                    or row.get("sentence", "")
                ).strip()

                audio = (
                    row.get("filename", "")
                    or row.get("audio", "")
                    or row.get("path", "")
                ).strip()

                if text and audio:
                    audio_path = data_dir / audio
                    if not audio_path.exists():
                        for subdir in ["wavs", "wav", "audio"]:
                            candidate = (
                                data_dir / subdir / audio
                            )
                            if candidate.exists():
                                audio_path = candidate
                                break

                    if audio_path.exists():
                        recordings.append({
                            "text": text.lower(),
                            "audio_path": str(audio_path),
                            "original_text": text,
                        })

        if recordings:
            log.info(
                "Loaded %d recordings from CSV",
                len(recordings),
            )
            return recordings

    for json_name in ["metadata.json", "data.json",
                       "index.json"]:
        json_path = data_dir / json_name
        if json_path.exists():
            with json_path.open(encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                for item in data:
                    text = item.get("text", "").strip()
                    audio = item.get("audio", "").strip()
                    if text and audio:
                        audio_path = Path(audio)
                        if not audio_path.is_absolute():
                            audio_path = data_dir / audio
                        if audio_path.exists():
                            recordings.append({
                                "text": text.lower(),
                                "audio_path": str(audio_path),
                                "original_text": text,
                            })

            if recordings:
                return recordings

    wav_files = sorted(data_dir.rglob("*.wav"))
    for wav in wav_files:
        for ext in [".txt", ".lab", ".text"]:
            text_file = wav.with_suffix(ext)
            if text_file.exists():
                text = text_file.read_text(
                    encoding="utf-8"
                ).strip()
                if text:
                    recordings.append({
                        "text": text.lower(),
                        "audio_path": str(wav),
                        "original_text": text,
                    })
                break

    log.info(
        "Loaded %d recordings from file scan",
        len(recordings),
    )
    return recordings



def get_char_ngrams(text: str, n: int = 3) -> Counter:
    """
    Extracts character n-grams from text.
    Used for fuzzy matching.
    """
    text = text.lower().strip()
    ngrams: Counter = Counter()
    for i in range(len(text) - n + 1):
        ngrams[text[i:i+n]] += 1
    return ngrams


def similarity_score(text1: str, text2: str) -> float:
    """
    Computes similarity between two texts
    using character n-gram overlap.

    Returns value between 0 and 1.
    1 = identical, 0 = no overlap.
    """
    if not text1 or not text2:
        return 0.0

    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    word_overlap = len(words1 & words2) / max(
        len(words1 | words2), 1
    )

    ng1 = get_char_ngrams(text1, n=3)
    ng2 = get_char_ngrams(text2, n=3)

    intersection = sum(
        min(ng1[k], ng2[k])
        for k in ng1 if k in ng2
    )
    union = sum(ng1.values()) + sum(ng2.values())

    char_score = (
        2 * intersection / union if union > 0 else 0.0
    )

    return word_overlap * 0.5 + char_score * 0.5


def find_best_match(
    query: str,
    recordings: list[dict[str, Any]],
    min_confidence: float = 0.15,
) -> tuple[dict[str, Any] | None, float]:
    """
    Finds the recording whose transcription
    best matches the query text.

    Args:
        query: Input Ibibio text to match
        recordings: All loaded recordings
        min_confidence: Minimum score to accept

    Returns:
        Tuple of (best_match, confidence_score)
        Returns (None, 0.0) if no good match found
    """
    if not recordings:
        return None, 0.0

    query_clean = query.lower().strip()

    best_match = None
    best_score = 0.0

    for recording in recordings:
        score = similarity_score(
            query_clean,
            recording["text"],
        )
        if score > best_score:
            best_score = score
            best_match = recording

    if best_score >= min_confidence:
        return best_match, best_score
    return None, best_score


def find_word_matches(
    query: str,
    recordings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Finds recordings containing key words
    from the query. Used when full phrase
    matching fails — returns multiple short
    clips that together cover the query.

    Args:
        query: Input text
        recordings: All recordings

    Returns:
        List of matching recordings sorted
        by relevance
    """
    query_words = set(query.lower().split())
    scored: list[tuple[float, dict[str, Any]]] = []

    for recording in recordings:
        rec_words = set(recording["text"].split())
        overlap = len(query_words & rec_words)
        if overlap > 0:
            score = overlap / max(
                len(query_words), 1
            )
            scored.append((score, recording))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:3]]



def concatenate_audio_files(
    audio_paths: list[str],
    output_path: str,
    pause_ms: int = 200,
) -> str | None:
    """
    Joins multiple WAV files with short pauses
    between them. Used when we match multiple
    short clips instead of one full phrase.

    Args:
        audio_paths: List of WAV file paths
        output_path: Output WAV path
        pause_ms: Silence between clips in ms

    Returns:
        Path to output file or None
    """
    try:
        import wave
        import struct
        import numpy as np

        sample_rate = 16000
        all_audio = []

        silence = np.zeros(
            int(sample_rate * pause_ms / 1000),
            dtype=np.int16,
        )

        for audio_path in audio_paths:
            try:
                with wave.open(audio_path, "r") as wf:
                    sr = wf.getframerate()
                    frames = wf.readframes(
                        wf.getnframes()
                    )
                    audio = np.frombuffer(
                        frames, dtype=np.int16
                    )

                    if sr != sample_rate:
                        ratio = sample_rate / sr
                        new_len = int(len(audio) * ratio)
                        indices = np.linspace(
                            0, len(audio) - 1, new_len
                        ).astype(int)
                        audio = audio[indices]

                    all_audio.append(audio)
                    all_audio.append(silence)

            except Exception as e:
                log.warning(
                    "Could not read %s: %s",
                    audio_path, e,
                )
                continue

        if not all_audio:
            return None

        combined = np.concatenate(all_audio)

        with wave.open(output_path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(combined.tobytes())

        return output_path

    except Exception as e:
        log.error("Audio concatenation failed: %s", e)
        return None


_recordings_cache: list[dict[str, Any]] | None = None
_data_dir_cache: Path | None = None


def speak_ibibio(
    text: str,
    output_path: str = "ibibio_voice_output.wav",
    data_dir: str = "africanvoices_data",
    fallback_lang: str = "yo",
) -> tuple[str | None, str]:
    """
    Main Ibibio TTS function using real recordings.

    Strategy:
    1. Try full phrase match from LLSTI recordings
    2. Try word-level matching and concatenate
    3. Fall back to gTTS with Yoruba phonology

    Args:
        text: Ibibio text to synthesise
        output_path: Output WAV path
        data_dir: AfricanVoices data directory
        fallback_lang: gTTS language for fallback

    Returns:
        Tuple of (audio_path, method_used)
        method: 'real_voice' | 'concatenated' | 'gtts'
    """
    global _recordings_cache, _data_dir_cache

    if not text or not text.strip():
        return None, "empty"

    if _recordings_cache is None:
        ibb_dir = find_ibibio_data(data_dir)
        if ibb_dir:
            _data_dir_cache = ibb_dir
            _recordings_cache = load_ibibio_recordings(
                ibb_dir
            )
            log.info(
                "Loaded %d Ibibio recordings",
                len(_recordings_cache),
            )
        else:
            _recordings_cache = []
            log.warning(
                "AfricanVoices data not found at %s. "
                "Falling back to gTTS.",
                data_dir,
            )

    recordings = _recordings_cache

    if recordings:
        match, confidence = find_best_match(
            text, recordings
        )

        if match and confidence >= 0.4:
            src = match["audio_path"]
            try:
                import shutil
                shutil.copy2(src, output_path)
                log.info(
                    "Real voice match (%.2f): %s → %s",
                    confidence,
                    match["original_text"][:40],
                    output_path,
                )
                return output_path, "real_voice"
            except Exception as e:
                log.warning("Copy failed: %s", e)

    if recordings:
        word_matches = find_word_matches(
            text, recordings
        )
        if word_matches:
            audio_paths = [
                m["audio_path"] for m in word_matches
            ]
            result = concatenate_audio_files(
                audio_paths, output_path
            )
            if result:
                log.info(
                    "Concatenated %d clips for: %s",
                    len(audio_paths), text[:40],
                )
                return result, "concatenated"

    log.info(
        "No recording match for '%s'. Using gTTS.",
        text[:40],
    )
    try:
        from gtts import gTTS
        mp3_path = output_path.replace(".wav", ".mp3")
        tts = gTTS(
            text=text,
            lang=fallback_lang,
            slow=False,
        )
        tts.save(mp3_path)
        log.info("gTTS fallback saved: %s", mp3_path)
        return mp3_path, "gtts"
    except Exception as e:
        log.error("gTTS fallback failed: %s", e)

    return None, "failed"


def generate_ibibio_phrase_library(
    phrases: list[tuple[str, str]],
    output_dir: str = "tts_output/demo_phrases",
    data_dir: str = "africanvoices_data",
) -> dict[str, dict[str, Any]]:
    """
    Generates audio for all demo phrases using
    real Ibibio voice where possible.

    Args:
        phrases: List of (ibibio_text, filename) tuples
        output_dir: Output directory
        data_dir: AfricanVoices data location

    Returns:
        Dict mapping filename to result info
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, Any]] = {}

    print("=" * 50)
    print("GENERATING IBIBIO PHRASE LIBRARY")
    print("Using real voice recordings where available")
    print("=" * 50)

    for phrase, name in phrases:
        output_path = str(out_dir / f"{name}.wav")

        audio_path, method = speak_ibibio(
            phrase,
            output_path=output_path,
            data_dir=data_dir,
        )

        method_label = {
            "real_voice":  "✓ Real Ibibio voice",
            "concatenated": "~ Concatenated clips",
            "gtts":         "○ gTTS fallback",
            "failed":       "✗ Failed",
        }.get(method, method)

        print(f"  {method_label}: '{phrase}'")

        results[name] = {
            "phrase": phrase,
            "audio_path": audio_path,
            "method": method,
        }

    real_count = sum(
        1 for r in results.values()
        if r["method"] == "real_voice"
    )
    concat_count = sum(
        1 for r in results.values()
        if r["method"] == "concatenated"
    )
    gtts_count = sum(
        1 for r in results.values()
        if r["method"] == "gtts"
    )

    print()
    print(f"  Real voice: {real_count}/{len(phrases)}")
    print(f"  Concatenated: {concat_count}/{len(phrases)}")
    print(f"  gTTS fallback: {gtts_count}/{len(phrases)}")
    print("=" * 50)

    return results



def inspect_dataset(
    data_dir: str = "africanvoices_data",
) -> None:
    """
    Inspects the downloaded AfricanVoices data
    and prints a summary of what was found.
    Run this after downloading the dataset.
    """
    print("=" * 50)
    print("AFRICANVOICES DATASET INSPECTION")
    print("=" * 50)

    base = Path(data_dir)
    if not base.exists():
        print(f"  ✗ Directory not found: {data_dir}")
        print(
            "  Download from: "
            "https://github.com/neulab/AfricanVoices"
        )
        return

    all_wav = list(base.rglob("*.wav"))
    all_txt = list(base.rglob("*.txt"))
    all_csv = list(base.rglob("*.csv"))
    all_json = list(base.rglob("*.json"))

    print(f"  WAV files: {len(all_wav)}")
    print(f"  TXT files: {len(all_txt)}")
    print(f"  CSV files: {len(all_csv)}")
    print(f"  JSON files: {len(all_json)}")
    print()

    print("  Folder structure:")
    for item in sorted(base.iterdir()):
        if item.is_dir():
            wav_count = len(list(item.rglob("*.wav")))
            print(
                f"    {item.name}/ "
                f"({wav_count} WAV files)"
            )

    print()

    ibb_dir = find_ibibio_data(data_dir)
    if ibb_dir:
        recordings = load_ibibio_recordings(ibb_dir)
        print(
            f"  ✓ Ibibio recordings found: "
            f"{len(recordings)}"
        )
        print(f"  Location: {ibb_dir}")
        print()
        print("  Sample recordings:")
        for rec in recordings[:5]:
            print(
                f"    {rec['original_text'][:60]}"
            )
    else:
        print("  ✗ No Ibibio recordings found")
        print(
            "  Look for an 'ibibio' or 'ibb' "
            "subfolder"
        )

    print("=" * 50)


if __name__ == "__main__":
    inspect_dataset("africanvoices_data")

    print()


    test_phrases = [
        ("ememe nnyin",              "greeting"),
        ("sosongo",                  "thank_you"),
        ("ami okut fo",              "love"),
        ("edisua edi mme ekpuk",     "education"),
        ("nyin edi nte",             "we_are_one"),
        ("abasi mbot fo",            "god_bless"),
        ("ndito edi iman",           "health_wealth"),
        ("eket nyin edi mme",        "our_language"),
        ("ami edi ke",               "i_am_here"),
        ("ke edi mme",               "it_is_good"),
    ]

    results = generate_ibibio_phrase_library(
        test_phrases,
        output_dir="tts_output/demo_phrases",
        data_dir="africanvoices_data",
    )

    print()
    print("Check tts_output/demo_phrases/ for audio.")
    print(
        "Real voice files will sound like "
        "a native Ibibio speaker."
    )
