from transformers import AutoTokenizer, AutoModel
import torch
import json
import os
import re

def load_model():
    """
    Loads XLM-R model - this is the multilingual model
    that understands African languages.
    Downloading it the first time will take a few minutes.
    """
    print("Loading XLM-R multilingual model...")
    print("(First time download may take 5-10 minutes)")
    print("(After first download it loads instantly)")
    print()

    model_name = "xlm-roberta-base"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    
    model.eval()

    print("Model loaded successfully")
    return tokenizer, model


def clean_text(text):
    """
    Cleans and normalizes the transcript text.
    Removes extra spaces, special characters etc.
    """
    
    text = re.sub(r'\s+', ' ', text)

    
    text = re.sub(r'[^\w\s\.\,\!\?\-\']', '', text)

    
    text = text.strip()

    return text

def split_into_sentences(text, words_per_chunk=50):
    """
    Splits text into chunks of 50 words each.
    We use words instead of punctuation because
    speech recognition does not add full stops.
    """
    words = text.split()
    sentences = []
    for i in range(0, len(words), words_per_chunk):
        chunk = ' '.join(words[i:i + words_per_chunk])
        sentences.append(chunk)
    return sentences


def tokenize_text(text, tokenizer, max_length=512):
    """
    Converts text into tokens the model can understand.
    Think of tokens as the smallest units of meaning.
    """
    tokens = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding=True
    )
    return tokens


def get_embeddings(tokens, model):
    """
    Generates contextual embeddings from tokens.
    Embeddings are numerical representations of meaning.
    This is how the model understands language.
    """
    with torch.no_grad():
        outputs = model(**tokens)

    
    embeddings = outputs.last_hidden_state

    
    sentence_embedding = embeddings.mean(dim=1)

    return sentence_embedding


def process_text(transcript_path="transcript.json"):
    """
    Main function - processes the transcript
    and generates embeddings for each sentence.
    """
    print("="*50)
    print("TEXT PROCESSING MODULE")
    print("="*50)

    
    if not os.path.exists(transcript_path):
        print(f"Error: {transcript_path} not found")
        print("Make sure you ran audio_module.py first")
        return None

    print("Step 1: Loading transcript...")
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript_data = json.load(f)

    raw_text = transcript_data["transcript"]
    print(f"Transcript loaded: {len(raw_text)} characters")
    print()

    
    print("Step 2: Cleaning and normalizing text...")
    cleaned_text = clean_text(raw_text)
    print(f"Cleaned text: {len(cleaned_text)} characters")
    print()

    
    print("Step 3: Splitting into sentences...")
    sentences = split_into_sentences(cleaned_text)
    print(f"Total sentences: {len(sentences)}")
    print()

    
    print("Step 4: Loading multilingual model...")
    tokenizer, model = load_model()
    print()

    
    print("Step 5: Generating embeddings...")
    print("(This understands the meaning of each sentence)")
    print()

    processed_sentences = []

    for i, sentence in enumerate(sentences[:20]):
        
        print(f"Processing sentence {i+1}: "
              f"{sentence[:50]}...")

     
        tokens = tokenize_text(sentence, tokenizer)

        
        embedding = get_embeddings(tokens, model)

        processed_sentences.append({
            "sentence_id": i,
            "text": sentence,
            "token_count": tokens['input_ids'].shape[1],
            "embedding_shape": list(embedding.shape)
        })

    print()

    
    output_data = {
        "original_transcript": raw_text,
        "cleaned_text": cleaned_text,
        "total_sentences": len(sentences),
        "all_sentences": sentences,
        "processed_sample": processed_sentences
    }

    output_path = "processed_text.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    print("="*50)
    print("TEXT PROCESSING COMPLETE")
    print(f"Total sentences found: {len(sentences)}")
    print(f"Processed data saved to: {output_path}")
    print()
    print("SAMPLE SENTENCES:")
    print("-"*50)
    for i, s in enumerate(sentences[:5]):
        print(f"{i+1}. {s}")
    print("="*50)

    return output_data


if __name__ == "__main__":
    result = process_text("transcript.json")