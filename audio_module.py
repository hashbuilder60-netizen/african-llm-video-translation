import whisper
import os
import json
import wave
import numpy as np


def get_audio_info(audio_path):
    """
    Reads basic information about the audio file.
    """
    with wave.open(audio_path, 'r') as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        duration = frames / float(rate)
        channels = wav_file.getnchannels()

        print(f"Audio file: {audio_path}")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Sample rate: {rate} Hz")
        print(f"Channels: {channels}")

        return duration, rate


def transcribe_audio(audio_path,
                     language=None):
    """
    Transcribes audio using Whisper.
    language=None means Whisper auto detects
    the language automatically.
    This works for English and African languages.
    """
    print("="*50)
    print("AUDIO PROCESSING MODULE")
    print("(Powered by Whisper - Offline)")
    print("="*50)

    
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found")
        print("Run video_module.py first")
        return None

    
    print("Step 1: Reading audio information...")
    duration, rate = get_audio_info(audio_path)
    print()

    print("Step 2: Loading Whisper model...")
    print("(First time downloads ~1.5GB)")
    print("(After that loads instantly)")

    
    
    model = whisper.load_model("base")
    print("Whisper model loaded")
    print()


    print("Step 3: Transcribing audio...")
    print("(This works offline)")
    print("(Auto detecting language...)")
    print()

    result = model.transcribe(
        audio_path,
        language=language,
        verbose=False
    )


    transcript = result["text"]
    detected_language = result["language"]
    segments = result["segments"]

    print(f"Language detected: {detected_language}")
    print(f"Transcript length: {len(transcript)} chars")
    print()

    
    word_count = len(transcript.split())

    
    with open("transcript.txt", "w",
              encoding="utf-8") as f:
        f.write(transcript)

    
    transcript_data = {
        "audio_file": audio_path,
        "detected_language": detected_language,
        "duration_seconds": duration,
        "transcript": transcript,
        "word_count": word_count,
        "segments": [
            {
                "id": seg["id"],
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"]
            }
            for seg in segments
        ]
    }

    with open("transcript.json", "w",
              encoding="utf-8") as f:
        json.dump(
            transcript_data, f,
            indent=4,
            ensure_ascii=False
        )

    print("="*50)
    print("AUDIO PROCESSING COMPLETE")
    print(f"Language: {detected_language}")
    print(f"Words transcribed: {word_count}")
    print(f"Transcript saved to: transcript.txt")
    print(f"JSON saved to: transcript.json")
    print()
    print("TRANSCRIPT PREVIEW:")
    print("-"*50)
    print(transcript[:500])
    print("="*50)

    return transcript


def calculate_wer(reference, hypothesis):
    """
    Calculates Word Error Rate between
    reference text and hypothesis text.
    WER = (S + D + I) / N
    S = substitutions
    D = deletions
    I = insertions
    N = number of words in reference
    """
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    
    d = np.zeros(
        (len(ref_words) + 1,
         len(hyp_words) + 1),
        dtype=np.uint32
    )

    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion = d[i][j-1] + 1
                deletion = d[i-1][j] + 1
                d[i][j] = min(
                    substitution,
                    insertion,
                    deletion
                )

    wer = float(d[len(ref_words)][len(hyp_words)])
    wer = wer / float(len(ref_words))
    wer = wer * 100  

    return round(wer, 2)


def evaluate_wer(audio_path, reference_text,
                 language=None):
    """
    Transcribes audio and calculates WER
    against a reference transcript.
    This is for your Chapter 5 evaluation.
    """
    print("="*50)
    print("WER EVALUATION")
    print("="*50)

    
    hypothesis = transcribe_audio(
        audio_path, language
    )

    if hypothesis is None:
        return None


    wer = calculate_wer(reference_text, hypothesis)

    print()
    print("WER RESULTS:")
    print("-"*50)
    print(f"Reference words: "
          f"{len(reference_text.split())}")
    print(f"Hypothesis words: "
          f"{len(hypothesis.split())}")
    print(f"Word Error Rate: {wer}%")
    print()

    if wer < 20:
        quality = "Excellent"
    elif wer < 40:
        quality = "Good"
    elif wer < 60:
        quality = "Moderate"
    else:
        quality = "Poor"

    print(f"Quality Rating: {quality}")
    print("="*50)


    wer_results = {
        "audio_file": audio_path,
        "language": language or "auto",
        "wer_percentage": wer,
        "quality_rating": quality,
        "reference_word_count": len(
            reference_text.split()
        ),
        "hypothesis_word_count": len(
            hypothesis.split()
        )
    }

    with open("wer_results.json", "w",
              encoding="utf-8") as f:
        json.dump(
            wer_results, f,
            indent=4,
            ensure_ascii=False
        )

    print(f"WER results saved to: wer_results.json")

    return wer



if __name__ == "__main__":
    transcript = transcribe_audio("audio.wav")