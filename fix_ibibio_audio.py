"""
Quick fix — stop the phoneme beeps.

Regenerates the main Ibibio audio output using gTTS
instead of the synthetic phoneme engine. Auto-detects
which language gTTS actually supports (its voice list
is much smaller than NLLB's 200 languages — Yoruba,
Hausa and Igbo are not on it).

Does NOT touch your real IbibioVoice demo phrase clips.
No need to reprocess the video. Run this once.
"""

import json
from pathlib import Path
from gtts import gTTS
from gtts.lang import tts_langs

out_dir = Path("tts_output")
out_dir.mkdir(exist_ok=True)

seg_path = Path("segment_translations/ibibio_segments.json")
text = "Ememe nnyin. Sosongo. Ami okut fo. Edisua edi mme."

if seg_path.exists():
    with seg_path.open(encoding="utf-8") as f:
        segs = json.load(f)
    sample = [
        s.get("translation", "") for s in segs[:5]
        if s.get("translation", "").strip()
    ]
    if sample:
        text = " ".join(sample)

print("Generating replacement audio with gTTS...")
print(f"Text sample: {text[:80]}...")
print()

supported = tts_langs()
print(f"gTTS voice engine supports {len(supported)} languages")

candidates = ["sw", "af", "ha", "yo", "ig", "fr", "en"]
chosen = next((c for c in candidates if c in supported), "en")

print(f"Selected: {chosen} ({supported.get(chosen)})")
print()

tts = gTTS(text=text, lang=chosen, slow=False)
tts.save(str(out_dir / "ibibio_tts.mp3"))
print(f"Saved: tts_output/ibibio_tts.mp3 (using {chosen})")

old_wav = out_dir / "ibibio_tts.wav"
if old_wav.exists():
    old_wav.unlink()
    print("Removed old phoneme WAV (the beeping file)")

print()
print("Done. Refresh your browser tab — no need to")
print("reprocess the video or restart Streamlit.")
print()
print("Your real IbibioVoice phrase clips in")
print("tts_output/demo_phrases/ are untouched.")
