"""
Advanced Evaluation Module.

Discipline: Machine Learning / NLP Evaluation
Refine means here: measuring model quality across
multiple dimensions, not just surface-level matching.

Metrics implemented:
    1. BLEU  - n-gram precision (already had this)
    2. chrF  - character n-gram F-score
             (better for morphologically rich languages
              like Ibibio where words share roots)
    3. METEOR - considers synonyms and stemming
    4. WER   - Word Error Rate for ASR evaluation
    5. ChrF++ - chrF with word n-grams added
    6. Catastrophic Forgetting test
             (measures if fine-tuning on Ibibio
              degraded Yoruba/Hausa/Igbo performance)

Why chrF matters more than BLEU for Ibibio:
    BLEU requires exact word matches.
    Ibibio words share roots and affixes.
    chrF scores partial character matches,
    which better reflects actual translation quality
    in morphologically rich languages.

Author: African LLM Project
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("evaluation.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)




@dataclass
class EvaluationResult:
    """Stores all evaluation scores for one model."""

    model_name: str
    language: str
    bleu: float = 0.0
    chrf: float = 0.0
    chrf_plus: float = 0.0
    meteor: float = 0.0
    wer: float = 0.0
    sample_count: int = 0
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "language": self.language,
            "bleu": round(self.bleu, 2),
            "chrf": round(self.chrf, 2),
            "chrf_plus": round(self.chrf_plus, 2),
            "meteor": round(self.meteor, 2),
            "wer": round(self.wer, 2),
            "sample_count": self.sample_count,
            "notes": self.notes,
        }

    def summary_line(self) -> str:
        return (
            f"  {self.model_name:<30} "
            f"BLEU: {self.bleu:>6.2f}  "
            f"chrF: {self.chrf:>6.2f}  "
            f"METEOR: {self.meteor:>6.2f}  "
            f"WER: {self.wer:>6.2f}%"
        )


@dataclass
class CatastrophicForgettingReport:
    """
    Measures if fine-tuning on Ibibio
    degraded performance on other languages.

    This is a critical missing test in your paper.
    Most low-resource fine-tuning papers measure this.
    """

    language: str
    before_bleu: float
    after_bleu: float
    before_chrf: float
    after_chrf: float

    @property
    def bleu_degradation(self) -> float:
        return self.before_bleu - self.after_bleu

    @property
    def chrf_degradation(self) -> float:
        return self.before_chrf - self.after_chrf

    @property
    def is_catastrophic(self) -> bool:
        """
        Degradation > 5 BLEU points is
        considered catastrophic in literature.
        """
        return self.bleu_degradation > 5.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "before_bleu": round(self.before_bleu, 2),
            "after_bleu": round(self.after_bleu, 2),
            "bleu_degradation": round(
                self.bleu_degradation, 2
            ),
            "before_chrf": round(self.before_chrf, 2),
            "after_chrf": round(self.after_chrf, 2),
            "chrf_degradation": round(
                self.chrf_degradation, 2
            ),
            "is_catastrophic": self.is_catastrophic,
        }




def compute_bleu(
    references: list[str],
    hypotheses: list[str],
    max_n: int = 4,
) -> float:
    """
    Computes corpus-level BLEU score.

    Args:
        references: Ground truth translations
        hypotheses: Model output translations
        max_n: Maximum n-gram order (default 4)

    Returns:
        BLEU score 0-100
    """
    if not references or not hypotheses:
        return 0.0

    clipped_counts: Counter[tuple[str, ...]] = Counter()
    total_counts: Counter[tuple[str, ...]] = Counter()
    ref_length = 0
    hyp_length = 0

    for ref, hyp in zip(references, hypotheses):
        ref_tokens = ref.lower().split()
        hyp_tokens = hyp.lower().split()
        ref_length += len(ref_tokens)
        hyp_length += len(hyp_tokens)

        for n in range(1, max_n + 1):
            ref_ngrams = Counter(
                tuple(ref_tokens[i:i+n])
                for i in range(len(ref_tokens)-n+1)
            )
            hyp_ngrams = Counter(
                tuple(hyp_tokens[i:i+n])
                for i in range(len(hyp_tokens)-n+1)
            )
            for ngram, count in hyp_ngrams.items():
                clipped_counts[ngram] += min(
                    count, ref_ngrams.get(ngram, 0)
                )
                total_counts[ngram] += count

    precisions: list[float] = []
    for n in range(1, max_n + 1):
        n_clipped = sum(
            v for k, v in clipped_counts.items()
            if len(k) == n
        )
        n_total = sum(
            v for k, v in total_counts.items()
            if len(k) == n
        )
        if n_total == 0:
            precisions.append(0.0)
        else:
            precisions.append(n_clipped / n_total)

    if any(p == 0 for p in precisions):
        return 0.0

    log_avg = sum(math.log(p) for p in precisions) / max_n

    
    if hyp_length >= ref_length:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_length / max(hyp_length, 1))

    bleu = bp * math.exp(log_avg) * 100
    return round(bleu, 2)


def compute_chrf(
    references: list[str],
    hypotheses: list[str],
    char_order: int = 6,
    word_order: int = 0,
    beta: float = 2.0,
) -> float:
    """
    Computes chrF score — character n-gram F-score.

    WHY THIS MATTERS FOR IBIBIO:
    Ibibio words share roots. For example:
    - edi (is)
    - edisua (learning) — contains 'edi'
    - edidem (government/authority)
    
    BLEU would score these as completely wrong
    if the exact word doesn't match.
    chrF scores the shared character sequences
    which better reflects linguistic similarity.

    Args:
        references: Ground truth translations
        hypotheses: Model output translations
        char_order: Character n-gram order (default 6)
        word_order: Word n-gram order (0 = chrF,
                    2 = chrF++ which adds word grams)
        beta: Beta parameter (2.0 weights recall
              more than precision — better for
              translation evaluation)

    Returns:
        chrF score 0-100
    """
    if not references or not hypotheses:
        return 0.0

    total_f: float = 0.0

    for ref, hyp in zip(references, hypotheses):
        
        char_p, char_r = _ngram_precision_recall(
            ref, hyp, char_order, level="char"
        )

        if word_order > 0:
            word_p, word_r = _ngram_precision_recall(
                ref, hyp, word_order, level="word"
            )
            
            precision = (char_p + word_p) / 2
            recall = (char_r + word_r) / 2
        else:
            precision = char_p
            recall = char_r

        
        if precision + recall == 0:
            f_score = 0.0
        else:
            f_score = (
                (1 + beta**2) * precision * recall
                / (beta**2 * precision + recall)
            )

        total_f += f_score

    chrf = (total_f / len(references)) * 100
    return round(chrf, 2)


def _ngram_precision_recall(
    reference: str,
    hypothesis: str,
    n: int,
    level: str = "char",
) -> tuple[float, float]:
    """
    Helper: computes n-gram precision and recall.

    Args:
        reference: Reference string
        hypothesis: Hypothesis string
        n: N-gram order
        level: 'char' or 'word'

    Returns:
        Tuple of (precision, recall)
    """
    if level == "char":
        ref_units = list(reference.lower())
        hyp_units = list(hypothesis.lower())
    else:
        ref_units = reference.lower().split()
        hyp_units = hypothesis.lower().split()

    def get_ngrams(units: list[str]) -> Counter[str]:
        return Counter(
            "".join(units[i:i+n]) if level == "char"
            else " ".join(units[i:i+n])
            for i in range(len(units) - n + 1)
        )

    ref_ngrams = get_ngrams(ref_units)
    hyp_ngrams = get_ngrams(hyp_units)

    if not hyp_ngrams or not ref_ngrams:
        return 0.0, 0.0

    # Clipped matches
    matches = sum(
        min(count, ref_ngrams.get(ng, 0))
        for ng, count in hyp_ngrams.items()
    )

    precision = matches / sum(hyp_ngrams.values())
    recall = matches / sum(ref_ngrams.values())

    return precision, recall


def compute_meteor(
    references: list[str],
    hypotheses: list[str],
) -> float:
    """
    Computes METEOR score.

    Unlike BLEU, METEOR considers:
    - Exact word matches
    - Stemmed matches (partial word matches)
    - This is important for Ibibio where words
      share stems (edi, edisua, edidem all share 'edi')

    Args:
        references: Ground truth translations
        hypotheses: Model output translations

    Returns:
        METEOR score 0-100
    """
    if not references or not hypotheses:
        return 0.0

    scores: list[float] = []

    for ref, hyp in zip(references, hypotheses):
        ref_tokens = ref.lower().split()
        hyp_tokens = hyp.lower().split()

        if not ref_tokens or not hyp_tokens:
            scores.append(0.0)
            continue

        
        ref_counts = Counter(ref_tokens)
        hyp_counts = Counter(hyp_tokens)
        exact_matches = sum(
            min(c, ref_counts.get(w, 0))
            for w, c in hyp_counts.items()
        )

        stem_matches = 0
        ref_stems = {w[:3]: w for w in ref_tokens}
        for hyp_word in hyp_tokens:
            stem = hyp_word[:3]
            if (stem in ref_stems and
                    ref_stems[stem] != hyp_word):
                stem_matches += 0.5  # Partial credit

        total_matches = exact_matches + stem_matches

        precision = total_matches / max(len(hyp_tokens), 1)
        recall = total_matches / max(len(ref_tokens), 1)

        if precision + recall == 0:
            scores.append(0.0)
            continue

        alpha = 0.9
        f_mean = (
            precision * recall
            / (alpha * precision + (1 - alpha) * recall)
        )

        chunks = _count_chunks(ref_tokens, hyp_tokens)
        penalty = 0.5 * (
            chunks / max(total_matches, 1)
        ) ** 3

        meteor = f_mean * (1 - penalty)
        scores.append(max(meteor, 0.0))

    result = (sum(scores) / len(scores)) * 100
    return round(result, 2)


def _count_chunks(
    ref_tokens: list[str],
    hyp_tokens: list[str],
) -> int:
    """
    Counts number of contiguous matched chunks.
    Used for METEOR chunk penalty.
    """
    ref_set = set(ref_tokens)
    chunks = 0
    in_chunk = False

    for token in hyp_tokens:
        if token in ref_set:
            if not in_chunk:
                chunks += 1
                in_chunk = True
        else:
            in_chunk = False

    return max(chunks, 1)


def compute_wer(
    reference: str,
    hypothesis: str,
) -> float:
    """
    Computes Word Error Rate.

    WER = (Substitutions + Deletions + Insertions)
          / Number of words in reference

    Used to evaluate speech recognition quality.
    Your paper promised this metric in Chapter 5.

    Args:
        reference: Correct transcript
        hypothesis: ASR output

    Returns:
        WER as percentage 0-100
    """
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    if not ref_words:
        return 0.0

    n = len(ref_words)
    m = len(hyp_words)

    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j-1],  
                    dp[i-1][j],    
                    dp[i][j-1],    
                )

    wer = (dp[n][m] / n) * 100
    return round(wer, 2)


TEST_PAIRS: list[dict[str, str]] = [
    {
        "english":   "Good morning",
        "reference": "Ememe nnyin",
    },
    {
        "english":   "Thank you",
        "reference": "Sosongo",
    },
    {
        "english":   "How are you",
        "reference": "Afo ke",
    },
    {
        "english":   "I am fine",
        "reference": "Ami adi mme",
    },
    {
        "english":   "God bless you",
        "reference": "Abasi mbot fo",
    },
    {
        "english":   "Education is important",
        "reference": "Edisua edi mme ekpuk",
    },
    {
        "english":   "Health is wealth",
        "reference": "Ndito edi iman",
    },
    {
        "english":   "I love my family",
        "reference": "Ami okut ufok ami",
    },
    {
        "english":   "We are one people",
        "reference": "Nyin edi nte",
    },
    {
        "english":   "Our language is beautiful",
        "reference": "Eket nyin edi mme",
    },
]

YORUBA_TEST: list[dict[str, str]] = [
    {"english": "Good morning",   "reference": "Ẹ káàárọ̀"},
    {"english": "Thank you",      "reference": "E se"},
    {"english": "How are you",    "reference": "Báwo ni"},
    {"english": "I am fine",      "reference": "Mo wa dada"},
    {"english": "God bless you",  "reference": "Ọlọrun bukun fun ọ"},
]

HAUSA_TEST: list[dict[str, str]] = [
    {"english": "Good morning",   "reference": "Ina kwana"},
    {"english": "Thank you",      "reference": "Na gode"},
    {"english": "How are you",    "reference": "Yaya kake"},
    {"english": "I am fine",      "reference": "Lafiya lau"},
    {"english": "God bless you",  "reference": "Allah ya albarka ka"},
]



def translate_batch(
    sentences: list[str],
    model_path: str,
    target_lang_code: str = "yor_Latn",
) -> list[str]:
    """
    Translates a batch of sentences using a model.

    Args:
        sentences: English sentences to translate
        model_path: Path to model or HF model name
        target_lang_code: NLLB language code

    Returns:
        List of translated strings
    """
    try:
        import torch
        from transformers import (
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
        )
    except ImportError:
        log.error("transformers not installed")
        return [""] * len(sentences)

    log.info("Loading model: %s", model_path)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path
        )
        model.eval()
    except Exception as e:
        log.error("Model load failed: %s", e)
        return [""] * len(sentences)

    target_id = tokenizer.convert_tokens_to_ids(
        target_lang_code
    )

    translations: list[str] = []

    for sentence in sentences:
        try:
            inputs = tokenizer(
                sentence,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            )
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    forced_bos_token_id=target_id,
                    max_length=128,
                    num_beams=4,
                    early_stopping=True,
                )
            translation = tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True,
            )[0]
            translations.append(translation)
        except Exception as e:
            log.warning(
                "Translation failed for '%s': %s",
                sentence, e,
            )
            translations.append("")

    return translations



def evaluate_model(
    model_path: str,
    test_pairs: list[dict[str, str]],
    model_name: str,
    language: str,
    target_lang_code: str = "yor_Latn",
) -> EvaluationResult:
    """
    Runs full evaluation suite on one model.

    Args:
        model_path: HF model name or local path
        test_pairs: List of english/reference dicts
        model_name: Human-readable name for report
        language: Target language name
        target_lang_code: NLLB language code

    Returns:
        EvaluationResult with all scores
    """
    log.info("Evaluating: %s on %s", model_name, language)

    english = [p["english"] for p in test_pairs]
    references = [p["reference"] for p in test_pairs]

    hypotheses = translate_batch(
        english, model_path, target_lang_code
    )

    result = EvaluationResult(
        model_name=model_name,
        language=language,
        sample_count=len(test_pairs),
    )

    result.bleu = compute_bleu(references, hypotheses)
    result.chrf = compute_chrf(
        references, hypotheses, word_order=0
    )
    result.chrf_plus = compute_chrf(
        references, hypotheses, word_order=2
    )
    result.meteor = compute_meteor(references, hypotheses)

    log.info(
        "%s → BLEU: %.2f  chrF: %.2f  METEOR: %.2f",
        model_name, result.bleu, result.chrf, result.meteor,
    )

    print(f"\n  Sample translations ({model_name}):")
    print(f"  {'English':<25} {'Reference':<25} {'Output'}")
    print("  " + "-" * 75)
    for en, ref, hyp in zip(
        english[:5], references[:5], hypotheses[:5]
    ):
        print(
            f"  {en:<25} {ref:<25} {hyp}"
        )

    return result


def check_catastrophic_forgetting(
    base_model: str,
    finetuned_model: str,
) -> list[CatastrophicForgettingReport]:
    """
    Checks if fine-tuning on Ibibio degraded
    performance on Yoruba and Hausa.

    This is a standard requirement in low-resource
    NLP papers. Your paper is missing this test.

    Args:
        base_model: Path to original NLLB-200
        finetuned_model: Path to your fine-tuned model

    Returns:
        List of forgetting reports per language
    """
    log.info("Checking catastrophic forgetting...")
    reports: list[CatastrophicForgettingReport] = []

    test_cases = [
        {
            "pairs": YORUBA_TEST,
            "language": "Yoruba",
            "code": "yor_Latn",
        },
        {
            "pairs": HAUSA_TEST,
            "language": "Hausa",
            "code": "hau_Latn",
        },
    ]

    for case in test_cases:
        english = [p["english"] for p in case["pairs"]]
        references = [
            p["reference"] for p in case["pairs"]
        ]

        base_hyp = translate_batch(
            english, base_model, case["code"]
        )
        before_bleu = compute_bleu(references, base_hyp)
        before_chrf = compute_chrf(references, base_hyp)

        ft_hyp = translate_batch(
            english, finetuned_model, case["code"]
        )
        after_bleu = compute_bleu(references, ft_hyp)
        after_chrf = compute_chrf(references, ft_hyp)

        report = CatastrophicForgettingReport(
            language=case["language"],
            before_bleu=before_bleu,
            after_bleu=after_bleu,
            before_chrf=before_chrf,
            after_chrf=after_chrf,
        )
        reports.append(report)

        status = "⚠️  CATASTROPHIC" if report.is_catastrophic else "✅  ACCEPTABLE"
        print(
            f"\n  {case['language']}: {status}"
            f"\n    BLEU: {before_bleu:.2f} → "
            f"{after_bleu:.2f} "
            f"(Δ {report.bleu_degradation:+.2f})"
            f"\n    chrF: {before_chrf:.2f} → "
            f"{after_chrf:.2f} "
            f"(Δ {report.chrf_degradation:+.2f})"
        )

    return reports


def run_full_evaluation() -> None:
    """
    Runs the complete evaluation pipeline.
    Generates Chapter 5 tables automatically.
    """
    print("=" * 55)
    print("ADVANCED EVALUATION SUITE")
    print("BLEU + chrF + chrF++ + METEOR + WER")
    print("=" * 55)

    all_results: list[EvaluationResult] = []

    print("\n[1/3] Evaluating baseline NLLB-200...")
    baseline = evaluate_model(
        model_path="facebook/nllb-200-distilled-600M",
        test_pairs=TEST_PAIRS,
        model_name="NLLB-200 Baseline",
        language="Ibibio",
        target_lang_code="yor_Latn",
    )
    all_results.append(baseline)

    print("\n[2/3] Evaluating fine-tuned Ibibio model...")
    finetuned_path = Path("ibibio_model")

    if finetuned_path.exists():
        finetuned = evaluate_model(
            model_path=str(finetuned_path),
            test_pairs=TEST_PAIRS,
            model_name="Fine-tuned Ibibio",
            language="Ibibio",
            target_lang_code="yor_Latn",
        )
        all_results.append(finetuned)
    else:
        log.warning(
            "ibibio_model not found. "
            "Run finetune_ibibio.py first."
        )
        finetuned = EvaluationResult(
            model_name="Fine-tuned Ibibio",
            language="Ibibio",
            notes="Model not found",
        )
        all_results.append(finetuned)

    print("\n[3/3] Checking catastrophic forgetting...")

    forgetting_reports: list[CatastrophicForgettingReport] = []
    if finetuned_path.exists():
        forgetting_reports = check_catastrophic_forgetting(
            base_model="facebook/nllb-200-distilled-600M",
            finetuned_model=str(finetuned_path),
        )
    else:
        print("  Skipped — fine-tuned model not found")

    print("\n── WER Demonstration ──")
    wer_examples = [
        {
            "reference": "education is important for every child",
            "hypothesis": "education is important for every child",
            "label": "Perfect transcription",
        },
        {
            "reference": "health is wealth",
            "hypothesis": "health is welth",
            "label": "One word error",
        },
        {
            "reference": "we are one people united",
            "hypothesis": "we one people",
            "label": "Missing words",
        },
    ]

    print(f"\n  {'Example':<28} {'WER %':>6}  Quality")
    print("  " + "-" * 50)
    wer_scores: list[float] = []
    for ex in wer_examples:
        wer = compute_wer(ex["reference"], ex["hypothesis"])
        wer_scores.append(wer)
        quality = (
            "Excellent" if wer < 10
            else "Good" if wer < 30
            else "Moderate" if wer < 60
            else "Poor"
        )
        print(f"  {ex['label']:<28} {wer:>6.1f}%  {quality}")

    avg_wer = sum(wer_scores) / len(wer_scores)

    print("\n" + "=" * 55)
    print("CHAPTER 5 EVALUATION TABLE")
    print("=" * 55)
    print(f"\n  {'Model':<30} "
          f"{'BLEU':>6}  "
          f"{'chrF':>6}  "
          f"{'METEOR':>6}  "
          f"{'WER%':>5}")
    print("  " + "-" * 60)
    for r in all_results:
        print(r.summary_line())

    if len(all_results) >= 2:
        bleu_gain = all_results[1].bleu - all_results[0].bleu
        chrf_gain = all_results[1].chrf - all_results[0].chrf
        print(f"\n  Improvement (Baseline → Fine-tuned):")
        print(f"    BLEU:   {bleu_gain:+.2f} points")
        print(f"    chrF:   {chrf_gain:+.2f} points")

    output = {
        "evaluation_results": [
            r.to_dict() for r in all_results
        ],
        "catastrophic_forgetting": [
            r.to_dict() for r in forgetting_reports
        ],
        "wer_examples": [
            {
                "label": ex["label"],
                "reference": ex["reference"],
                "hypothesis": ex["hypothesis"],
                "wer": wer,
            }
            for ex, wer in zip(wer_examples, wer_scores)
        ],
        "average_wer": round(avg_wer, 2),
        "metric_explanations": {
            "BLEU": (
                "Measures exact n-gram matches. "
                "Higher is better. Range 0-100."
            ),
            "chrF": (
                "Character n-gram F-score. Better than "
                "BLEU for morphologically rich languages "
                "like Ibibio. Higher is better."
            ),
            "chrF++": (
                "chrF with word n-grams added. "
                "Most robust metric for Ibibio."
            ),
            "METEOR": (
                "Considers synonyms and stemming. "
                "Better than BLEU for low-resource "
                "languages with shared word roots."
            ),
            "WER": (
                "Word Error Rate for speech recognition. "
                "Lower is better. Range 0-100%."
            ),
        },
    }

    report_path = Path("advanced_evaluation_results.json")
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nFull report saved to: {report_path}")
    print("\nCopy the table above into Chapter 5")
    print("Section 5.2 — Quantitative Evaluation")
    print("=" * 55)


if __name__ == "__main__":
    run_full_evaluation()
