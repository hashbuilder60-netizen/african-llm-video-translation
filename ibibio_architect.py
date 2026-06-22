import json
import re
import os
from collections import Counter, defaultdict
import numpy as np

print("="*50)
print("IBIBIO LINGUISTIC ARCHITECTURE BUILDER")
print("="*50)


print("\nStep 1: Loading Ibibio corpus...")

with open("corpus.json", "r", encoding="utf-8") as f:
    corpus = json.load(f)

english_sentences = [item["english"] for item in corpus]
ibibio_sentences = [item["ibibio"] for item in corpus]

print(f"Loaded {len(corpus)} sentence pairs")



print("\nStep 2: Building Ibibio phoneme inventory...")

def extract_phonemes(sentences):
    """
    Extracts all unique characters and 
    character combinations from Ibibio text.
    This builds the phonological profile.
    """
    all_chars = Counter()
    bigrams = Counter()
    trigrams = Counter()
    
    for sentence in sentences:
        sentence = sentence.lower()
        chars = list(sentence)
        
        
        for char in chars:
            if char.isalpha():
                all_chars[char] += 1
        
        
        for i in range(len(chars) - 1):
            if chars[i].isalpha() and chars[i+1].isalpha():
                bigrams[chars[i] + chars[i+1]] += 1
        
        
        for i in range(len(chars) - 2):
            if (chars[i].isalpha() and 
                chars[i+1].isalpha() and 
                chars[i+2].isalpha()):
                trigrams[''.join(chars[i:i+3])] += 1
    
    return all_chars, bigrams, trigrams

phonemes, bigrams, trigrams = extract_phonemes(
    ibibio_sentences
)

print(f"Unique phonemes found: {len(phonemes)}")
print(f"Bigram patterns: {len(bigrams)}")
print(f"Trigram patterns: {len(trigrams)}")
print(f"\nTop 10 phonemes: {phonemes.most_common(10)}")
print(f"Top 10 bigrams: {bigrams.most_common(10)}")



print("\nStep 3: Extracting Ibibio morphemes...")

def extract_morphemes(sentences):
    """
    Identifies root words, prefixes and suffixes
    in Ibibio by finding common substrings
    across multiple words.
    """
    all_words = []
    for sentence in sentences:
        words = sentence.lower().split()
        all_words.extend(words)
    
    word_freq = Counter(all_words)
    
    
    prefixes = Counter()
    suffixes = Counter()
    roots = Counter()
    
    for word in all_words:
        if len(word) >= 3:
            prefixes[word[:2]] += 1
            prefixes[word[:3]] += 1
            suffixes[word[-2:]] += 1
            suffixes[word[-3:]] += 1
            
            
            if len(word) >= 4:
                root = word[1:-1]
                roots[root] += 1
    
    return word_freq, prefixes, suffixes, roots

word_freq, prefixes, suffixes, roots = extract_morphemes(
    ibibio_sentences
)

print(f"Unique words: {len(word_freq)}")
print(f"Top 10 words: {word_freq.most_common(10)}")
print(f"Top prefixes: {prefixes.most_common(5)}")
print(f"Top suffixes: {suffixes.most_common(5)}")



print("\nStep 4: Analyzing syllable structure...")

VOWELS = set('aeiouáéíóúàèìòùâêîôû')

def get_syllable_pattern(word):
    """
    Returns C (consonant) V (vowel) pattern.
    Example: ibibio = V-CV-CV-V
    """
    pattern = []
    for char in word.lower():
        if char in VOWELS:
            pattern.append('V')
        elif char.isalpha():
            pattern.append('C')
    return '-'.join(pattern)

syllable_patterns = Counter()
for sentence in ibibio_sentences:
    for word in sentence.split():
        pattern = get_syllable_pattern(word)
        syllable_patterns[pattern] += 1

print("Top syllable patterns in Ibibio:")
for pattern, count in syllable_patterns.most_common(10):
    print(f"  {pattern}: {count} times")



print("\nAlgorithm 1: BPE (Byte Pair Encoding)...")

def build_bpe_vocabulary(sentences, num_merges=50):
    """
    Builds a BPE vocabulary from Ibibio text.
    BPE finds the most common character pairs
    and merges them into tokens.
    This creates subword units specific to Ibibio.
    """
    
    vocab = Counter()
    for sentence in sentences:
        for word in sentence.lower().split():
            
            word_chars = ' '.join(list(word)) + ' </w>'
            vocab[word_chars] += 1
    
    
    merges = []
    
    for i in range(num_merges):
        
        pairs = Counter()
        for word, freq in vocab.items():
            symbols = word.split()
            for j in range(len(symbols) - 1):
                pairs[(symbols[j], symbols[j+1])] += freq
        
        if not pairs:
            break
        
        
        best_pair = max(pairs, key=pairs.get)
        merges.append(best_pair)
        
        
        new_vocab = Counter()
        bigram = re.escape(' '.join(best_pair))
        replacement = ''.join(best_pair)
        
        for word, freq in vocab.items():
            new_word = re.sub(bigram, replacement, word)
            new_vocab[new_word] += freq
        
        vocab = new_vocab
    
    return vocab, merges

bpe_vocab, bpe_merges = build_bpe_vocabulary(
    ibibio_sentences, 
    num_merges=30
)

print(f"BPE merges performed: {len(bpe_merges)}")
print(f"Top 10 BPE merge rules:")
for pair in bpe_merges[:10]:
    print(f"  '{pair[0]}' + '{pair[1]}' → '{pair[0]+pair[1]}'")



print("\nAlgorithm 2: N-gram Language Model...")

def build_ngram_model(sentences, n=2):
    """
    Builds an n-gram model of Ibibio.
    This captures which words commonly follow 
    each other in Ibibio sentences.
    Used to score and validate translations.
    """
    ngrams = defaultdict(Counter)
    
    for sentence in sentences:
        words = ['<START>'] + sentence.lower().split() + ['<END>']
        
        for i in range(len(words) - n + 1):
            context = tuple(words[i:i+n-1])
            next_word = words[i+n-1]
            ngrams[context][next_word] += 1
    
    
    ngram_probs = {}
    for context, next_words in ngrams.items():
        total = sum(next_words.values())
        ngram_probs[context] = {
            word: count/total 
            for word, count in next_words.items()
        }
    
    return ngram_probs

bigram_model = build_ngram_model(ibibio_sentences, n=2)
trigram_model = build_ngram_model(ibibio_sentences, n=3)

print(f"Bigram contexts: {len(bigram_model)}")
print(f"Trigram contexts: {len(trigram_model)}")


print("\nSample Ibibio word predictions:")
for context, predictions in list(bigram_model.items())[:5]:
    if context[0] != '<START>':
        top_pred = max(predictions, key=predictions.get)
        prob = predictions[top_pred]
        print(f"  After '{context[0]}' → '{top_pred}' ({prob:.2f})")



print("\nAlgorithm 3: Phonological Similarity Scoring...")


yoruba_ibibio_pairs = [
    ("omi", "mmong", "water"),
    ("eko", "edisua", "learning"),
    ("ile", "ufok", "home"),
    ("orun", "abasi", "sky/god"),
    ("enia", "mme nte", "people"),
    ("ojo", "mmong", "rain"),
    ("omo", "emi", "child"),
    ("agba", "nnyin", "elder"),
    ("ogo", "iman", "wealth/glory"),
    ("ife", "okut", "love"),
]

def phonological_similarity(word1, word2):
    """
    Calculates how phonologically similar
    two words are based on shared characters,
    bigrams and syllable structure.
    Higher score = more similar.
    """
    word1, word2 = word1.lower(), word2.lower()
    
    
    chars1 = set(word1)
    chars2 = set(word2)
    char_overlap = len(chars1 & chars2) / max(
        len(chars1 | chars2), 1
    )

    
    bigrams1 = set(word1[i:i+2] for i in range(len(word1)-1))
    bigrams2 = set(word2[i:i+2] for i in range(len(word2)-1))
    bigram_overlap = len(bigrams1 & bigrams2) / max(
        len(bigrams1 | bigrams2), 1
    )
    

    len_sim = 1 - abs(len(word1) - len(word2)) / max(
        len(word1), len(word2), 1
    )
    
    
    pat1 = get_syllable_pattern(word1)
    pat2 = get_syllable_pattern(word2)
    pat_sim = 1.0 if pat1 == pat2 else 0.0
    

    score = (
        char_overlap * 0.3 +
        bigram_overlap * 0.3 +
        len_sim * 0.2 +
        pat_sim * 0.2
    )
    
    return round(score, 3)

print("\nYoruba-Ibibio phonological similarity scores:")
print(f"{'English':<15} {'Yoruba':<12} {'Ibibio':<15} {'Score'}")
print("-"*50)
for yoruba, ibibio, english in yoruba_ibibio_pairs:
    score = phonological_similarity(yoruba, ibibio)
    print(f"{english:<15} {yoruba:<12} {ibibio:<15} {score}")



print("\nAlgorithm 4: Cross-lingual Vocabulary Interpolation...")

def interpolate_vocabularies(
    ibibio_words, 
    yoruba_words, 
    efik_words,
    alpha=0.6, 
    beta=0.25,   
    gamma=0.15   
):
    """
    Creates an interpolated vocabulary by
    weighting Ibibio, Efik and Yoruba words
    based on their frequency and similarity.
    
    Efik is weighted higher than Yoruba because
    Efik and Ibibio are from the same language family
    (Cross River languages) while Yoruba is not.
    
    alpha + beta + gamma must equal 1.0
    """
    assert abs(alpha + beta + gamma - 1.0) < 0.01
    
    interpolated = {}
    
    all_meanings = set(
        list(ibibio_words.keys()) + 
        list(efik_words.keys()) + 
        list(yoruba_words.keys())
    )
    
    for meaning in all_meanings:
        ibb = ibibio_words.get(meaning, "")
        efik = efik_words.get(meaning, "")
        yor = yoruba_words.get(meaning, "")
        
        
        scores = {}
        if ibb:
            scores[ibb] = alpha
        if efik:
            efik_score = beta + (
                phonological_similarity(ibb, efik) * 0.1
                if ibb else 0
            )
            scores[efik] = efik_score
        if yor:
            yor_score = gamma + (
                phonological_similarity(ibb, yor) * 0.05
                if ibb else 0
            )
            if yor in scores:
                scores[yor] += yor_score
            else:
                scores[yor] = yor_score
        

        if scores:
            best = max(scores, key=scores.get)
            interpolated[meaning] = {
                "best": best,
                "ibibio": ibb,
                "efik": efik,
                "yoruba": yor,
                "confidence": round(
                    scores.get(best, 0), 3
                )
            }
    
    return interpolated


ibibio_vocab = {
    "good morning": "Ememe nnyin",
    "thank you": "Sosongo",
    "god": "Abasi",
    "water": "mmong",
    "house": "ufok",
    "child": "emi",
    "elder": "nnyin",
    "love": "okut",
    "learning": "edisua",
    "people": "mme nte"
}

efik_vocab = {
    "good morning": "Ememe",
    "thank you": "Sosongo",
    "god": "Abasi",
    "water": "mmong",
    "house": "ufok",
    "child": "emi",
    "elder": "ndidem",
    "love": "ke",
    "learning": "isua",
    "people": "mme"
}

yoruba_vocab = {
    "good morning": "Ẹ káàárọ̀",
    "thank you": "E se",
    "god": "Olorun",
    "water": "omi",
    "house": "ile",
    "child": "omo",
    "elder": "agba",
    "love": "ife",
    "learning": "eko",
    "people": "enia"
}

interpolated = interpolate_vocabularies(
    ibibio_vocab,
    yoruba_vocab,
    efik_vocab
)

print("\nInterpolated vocabulary results:")
print(f"{'Meaning':<15} {'Best':<20} {'Confidence'}")
print("-"*50)
for meaning, data in interpolated.items():
    print(
        f"{meaning:<15} "
        f"{data['best']:<20} "
        f"{data['confidence']}"
    )



print("\nStep 5: Saving Ibibio linguistic architecture...")

architecture = {
    "phoneme_inventory": dict(phonemes.most_common()),
    "top_bigrams": dict(bigrams.most_common(20)),
    "top_trigrams": dict(trigrams.most_common(20)),
    "syllable_patterns": dict(
        syllable_patterns.most_common(10)
    ),
    "word_frequency": dict(word_freq.most_common(50)),
    "top_prefixes": dict(prefixes.most_common(10)),
    "top_suffixes": dict(suffixes.most_common(10)),
    "bpe_merges": [
        list(pair) for pair in bpe_merges
    ],
    "interpolated_vocabulary": interpolated,
    "statistics": {
        "total_sentences": len(ibibio_sentences),
        "unique_words": len(word_freq),
        "unique_phonemes": len(phonemes),
        "bpe_merges_performed": len(bpe_merges),
        "syllable_pattern_types": len(syllable_patterns)
    }
}

with open("ibibio_architecture.json", "w",
          encoding="utf-8") as f:
    json.dump(
        architecture, f,
        indent=4,
        ensure_ascii=False
    )

print("\n" + "="*50)
print("LINGUISTIC ARCHITECTURE COMPLETE")
print("="*50)
print(f"\nStatistics:")
print(f"  Unique phonemes: {len(phonemes)}")
print(f"  Unique words: {len(word_freq)}")
print(f"  Syllable patterns: {len(syllable_patterns)}")
print(f"  BPE merge rules: {len(bpe_merges)}")
print(f"  Interpolated vocab entries: {len(interpolated)}")
print(f"\nSaved to: ibibio_architecture.json")
print("\nUse this architecture to:")
print("  1. Guide corpus expansion")
print("  2. Validate translations")
print("  3. Build better tokenizer")
print("  4. Document in Chapter 4")
print("="*50)