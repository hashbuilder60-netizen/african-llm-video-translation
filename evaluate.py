import json
import sacrebleu
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

print("="*50)
print("EVALUATION AND BLEU SCORES")
print("="*50)


def calculate_bleu(references, hypotheses):
    """
    Calculates BLEU score between
    reference and hypothesis translations.
    """
    bleu = sacrebleu.corpus_bleu(
        hypotheses,
        [references]
    )
    return bleu.score


def load_ibibio_model():
    """Loads the fine tuned Ibibio model."""
    print("Loading fine-tuned Ibibio model...")
    tokenizer = AutoTokenizer.from_pretrained(
        "ibibio_model"
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "ibibio_model"
    )
    model.eval()
    print("Ibibio model loaded")
    return tokenizer, model


def load_nllb_model():
    """Loads the original NLLB-200 model."""
    print("Loading original NLLB-200 model...")
    tokenizer = AutoTokenizer.from_pretrained(
        "facebook/nllb-200-distilled-600M"
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "facebook/nllb-200-distilled-600M"
    )
    model.eval()
    print("NLLB-200 model loaded")
    return tokenizer, model


def translate(text, tokenizer, model,
              target_lang="yor_Latn"):
    """Translates text using given model."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )

    target_id = tokenizer.convert_tokens_to_ids(
        target_lang
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=target_id,
            max_length=128,
            num_beams=4,
            early_stopping=True
        )

    return tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True
    )[0]



test_pairs = [
    ("Good morning", "Ememe nnyin"),
    ("Thank you", "Sosongo"),
    ("How are you", "Afo ke"),
    ("I am fine", "Ami adi mme"),
    ("God bless you", "Abasi mbot fo"),
    ("Education is important", "Edisua edi mme ekpuk"),
    ("Health is wealth", "Ndito edi iman"),
    ("I love my family", "Ami okut ufok ami"),
    ("We are one people", "Nyin edi nte"),
    ("Our language is beautiful", "Eket nyin edi mme"),
]

english_sentences = [p[0] for p in test_pairs]
reference_translations = [p[1] for p in test_pairs]

print()
print("Step 1: Evaluating fine-tuned Ibibio model...")
ibibio_tokenizer, ibibio_model = load_ibibio_model()
print()

ibibio_hypotheses = []
for sentence in english_sentences:
    translation = translate(
        sentence,
        ibibio_tokenizer,
        ibibio_model
    )
    ibibio_hypotheses.append(translation)

ibibio_bleu = calculate_bleu(
    reference_translations,
    ibibio_hypotheses
)

print()
print("Step 2: Evaluating baseline NLLB-200 model...")
nllb_tokenizer, nllb_model = load_nllb_model()
print()

nllb_hypotheses = []
for sentence in english_sentences:
    translation = translate(
        sentence,
        nllb_tokenizer,
        nllb_model
    )
    nllb_hypotheses.append(translation)

nllb_bleu = calculate_bleu(
    reference_translations,
    nllb_hypotheses
)

print()
print("="*50)
print("EVALUATION RESULTS")
print("="*50)
print()
print("BLEU SCORES:")
print(f"  Baseline NLLB-200: {nllb_bleu:.2f}")
print(f"  Fine-tuned Ibibio: {ibibio_bleu:.2f}")
print()

improvement = ibibio_bleu - nllb_bleu
if improvement > 0:
    print(f"  Improvement: +{improvement:.2f} BLEU points")
else:
    print(f"  Difference: {improvement:.2f} BLEU points")

print()
print("TRANSLATION COMPARISON:")
print("-"*50)

results = []
for i, (english, reference) in enumerate(test_pairs):
    print(f"English:   {english}")
    print(f"Reference: {reference}")
    print(f"Baseline:  {nllb_hypotheses[i]}")
    print(f"Fine-tuned:{ibibio_hypotheses[i]}")
    print()
    results.append({
        "english": english,
        "reference": reference,
        "baseline_nllb": nllb_hypotheses[i],
        "finetuned_ibibio": ibibio_hypotheses[i]
    })


evaluation_output = {
    "bleu_scores": {
        "baseline_nllb200": round(nllb_bleu, 2),
        "finetuned_ibibio": round(ibibio_bleu, 2),
        "improvement": round(improvement, 2)
    },
    "translation_comparisons": results,
    "test_sentences": len(test_pairs),
    "corpus_size": 226
}

with open("evaluation_results.json", "w",
          encoding="utf-8") as f:
    json.dump(
        evaluation_output, f,
        indent=4,
        ensure_ascii=False
    )

print("="*50)
print("EVALUATION COMPLETE")
print(f"Results saved to: evaluation_results.json")
print()
print("Use these BLEU scores in your Chapter 5")
print("="*50)