import json
import os
import torch
from transformers import AutoTokenizer, AutoModel


def load_text_model():
    print("Loading language model...")
    tokenizer = AutoTokenizer.from_pretrained(
        "xlm-roberta-base"
    )
    model = AutoModel.from_pretrained(
        "xlm-roberta-base"
    )
    model.eval()
    print("Language model loaded")
    return tokenizer, model


def summarize_text(text):
    """
    Extracts the most important sentences.
    No extra model needed.
    """
    words = text.split()
    sentences = []
    for i in range(0, len(words), 50):
        chunk = ' '.join(words[i:i + 50])
        sentences.append(chunk)

    
    word_scores = {}
    for sentence in sentences:
        for word in sentence.lower().split():
            word = word.strip('.,!?-\'\"')
            if len(word) > 4:
                word_scores[word] = (
                    word_scores.get(word, 0) + 1
                )

    
    scored = []
    for i, sentence in enumerate(sentences):
        score = 0
        for word in sentence.lower().split():
            word = word.strip('.,!?-\'\"')
            if word in word_scores:
                score += word_scores[word]
        scored.append((score, i, sentence))

    
    scored.sort(reverse=True)
    top = scored[:5]
    top.sort(key=lambda x: x[1])

    return ' '.join([s[2] for s in top])


def extract_key_topics(sentences):
    """
    Finds the most repeated meaningful words.
    """
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but',
        'in', 'on', 'at', 'to', 'for', 'of',
        'with', 'by', 'from', 'is', 'are', 'was',
        'were', 'be', 'been', 'have', 'has', 'had',
        'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'can', 'this', 'that',
        'these', 'those', 'it', 'its', 'you', 'your',
        'i', 'my', 'we', 'our', 'they', 'their',
        'what', 'how', 'when', 'where', 'why', 'who',
        'so', 'if', 'not', 'just', 'also', 'very',
        'really', 'about', 'up', 'out', 'all', 'one'
    }

    word_freq = {}
    for sentence in sentences:
        for word in sentence.lower().split():
            word = word.strip('.,!?-\'\"')
            if len(word) > 3 and word not in stop_words:
                word_freq[word] = (
                    word_freq.get(word, 0) + 1
                )

    sorted_topics = sorted(
        word_freq.items(),
        key=lambda x: x[1],
        reverse=True
    )
    return [word for word, freq in sorted_topics[:10]]


def process_with_transformer():
    print("="*50)
    print("TRANSFORMER BACKBONE MODULE")
    print("="*50)

    
    print("Step 1: Loading fused data...")
    if not os.path.exists("fused_data.json"):
        print("Error: Run fusion_module.py first")
        return None

    with open("fused_data.json", "r",
              encoding="utf-8") as f:
        fused_data = json.load(f)
    print(f"Loaded {fused_data['total_segments']} "
          f"segments")

    
    print("Step 2: Loading full transcript...")
    with open("processed_text.json", "r",
              encoding="utf-8") as f:
        text_data = json.load(f)

    full_text = text_data["cleaned_text"]
    all_sentences = text_data["all_sentences"]
    print(f"Full text: {len(full_text)} characters")
    print()

    
    print("Step 3: Extracting key topics...")
    topics = extract_key_topics(all_sentences)
    print(f"Key topics: {', '.join(topics)}")
    print()

    
    print("Step 4: Generating video summary...")
    summary = summarize_text(full_text)
    print("Summary generated")
    print()

    
    print("Step 5: Analyzing each segment...")
    segment_analysis = []
    for segment in fused_data["fused_segments"]:
        analysis = {
            "segment_id": segment["segment_id"],
            "timestamp": segment["timestamp"],
            "text": segment["text"],
            "word_count": len(
                segment["text"].split()
            ),
            "has_visual": segment[
                "fusion_summary"]["has_visual"],
            "frames_count": segment[
                "fusion_summary"]["frames_analyzed"]
        }
        segment_analysis.append(analysis)
        print(
            f"  Segment {segment['segment_id']+1} "
            f"at {segment['timestamp']}: "
            f"{len(segment['text'].split())} words"
        )
    print()

    
    output = {
        "video_summary": summary,
        "key_topics": topics,
        "total_segments": len(segment_analysis),
        "segment_analysis": segment_analysis,
        "statistics": {
            "total_words": len(full_text.split()),
            "total_sentences": len(all_sentences),
            "video_duration": fused_data[
                "video_duration"]
        }
    }

    output_path = "transformer_output.json"
    with open(output_path, "w",
              encoding="utf-8") as f:
        json.dump(output, f, indent=4,
                  ensure_ascii=False)

    print("="*50)
    print("TRANSFORMER BACKBONE COMPLETE")
    print()
    print("VIDEO SUMMARY:")
    print("-"*50)
    print(summary)
    print()
    print("KEY TOPICS:")
    print("-"*50)
    print(', '.join(topics))
    print()
    print(f"Results saved to: {output_path}")
    print("="*50)

    return output


if __name__ == "__main__":
    process_with_transformer()