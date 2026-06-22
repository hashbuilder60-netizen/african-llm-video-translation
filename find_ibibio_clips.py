import csv
import shutil
from pathlib import Path


METADATA_FILE = r"C:\Users\hashb\Downloads\IbibioVoice-main\IbibioVoice-main\validated.tsv"
CLIPS_FOLDER  = r"C:\Users\hashb\Downloads\IbibioVoice-main\IbibioVoice-main\clips"
OUTPUT_FOLDER = r"C:\Users\hashb\african_llm\venv\tts_output\demo_phrases"

Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)


TARGET_PHRASES = {
    "amesiere":   "greeting",
    "mmokom":     "greetings",
    "amedi":      "welcome",
    "sosongo":    "thank_you",
    "ami okut":   "love",
    "abasi":      "god_bless",
    "nyin edi":   "we_are_one",
    "ndito":      "health_wealth",
    "edisua":     "education",
    "ememe":      "good_morning",
}

found = 0
with open(METADATA_FILE, encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        sentence = row.get("sentence", "").lower()
        filename = row.get("path", "")

        for keyword, name in TARGET_PHRASES.items():
            if keyword in sentence:
                src = Path(CLIPS_FOLDER) / filename
                dst = Path(OUTPUT_FOLDER) / f"{name}.mp3"
                if src.exists() and not dst.exists():
                    shutil.copy2(src, dst)
                    print(f"✓ {name}: '{sentence[:50]}'")
                    found += 1
                    break

print(f"\nCopied {found} real Ibibio voice files")
print(f"To: {OUTPUT_FOLDER}")