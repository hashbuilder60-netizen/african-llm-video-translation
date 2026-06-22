import json
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


LANGUAGES = {
    "yoruba": "yor_Latn",
    "hausa": "hau_Latn",
    "igbo": "ibo_Latn",
}


def load_translation_model():
    """
    Loads the NLLB-200 translation model.
    This model supports Yoruba, Hausa and Igbo.
    First download is about 1.2GB.
    """
    print("Loading NLLB-200 translation model...")
    print("(First download may take 10-15 minutes)")
    print("(Supports 200 languages including African ones)")

    model_name = "facebook/nllb-200-distilled-600M"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name
    )
    model.eval()

    print("Translation model loaded successfully")
    return tokenizer, model


def translate_text(text, target_language,
                   tokenizer, model,
                   source_language="eng_Latn"):
    """
    Translates text from English to target language.
    """
    
    lang_code = LANGUAGES.get(
        target_language.lower(),
        target_language
    )

    try:
        
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

    
        target_lang_id = tokenizer.convert_tokens_to_ids(
            lang_code
        )

        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=target_lang_id,
                max_length=512,
                num_beams=4,
                early_stopping=True
            )

        
        translation = tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True
        )[0]

        return translation

    except Exception as e:
        print(f"  Translation error: {e}")
        return f"Translation failed: {str(e)}"


def translate_to_all_languages(text, tokenizer, model):
    """
    Translates text into all target African languages.
    """
    results = {"original_english": text}

    for language in LANGUAGES.keys():
        print(f"  Translating to {language.capitalize()}...")
        translation = translate_text(
            text,
            language,
            tokenizer,
            model
        )
        results[language] = translation
        print(f"  Done: {translation[:80]}...")

    return results


def run_translation():
    """
    Main translation function.
    """
    print("="*50)
    print("TRANSLATION OUTPUT MODULE")
    print("="*50)

    
    print("Step 1: Loading video summary...")
    if not os.path.exists("transformer_output.json"):
        print("Error: transformer_output.json not found")
        print("Run transformer_module.py first")
        return None

    with open("transformer_output.json", "r",
              encoding="utf-8") as f:
        transformer_data = json.load(f)

    summary = transformer_data["video_summary"]
    topics = transformer_data["key_topics"]

    print(f"Summary loaded: {len(summary)} characters")
    print(f"Key topics: {', '.join(topics[:5])}")
    print()

    
    print("Step 2: Loading translation model...")
    tokenizer, model = load_translation_model()
    print()

    
    print("Step 3: Translating video summary...")
    print("(This may take a few minutes per language)")
    print()

    
    
    short_summary = ' '.join(summary.split()[:200])
    translations = translate_to_all_languages(
        short_summary,
        tokenizer,
        model
    )
    print()

    
    print("Step 4: Translating key topics...")
    topics_text = "Key topics: " + ", ".join(topics)
    topic_translations = translate_to_all_languages(
        topics_text,
        tokenizer,
        model
    )
    print()

    
    output = {
        "original_summary": summary,
        "translated_summary": translations,
        "original_topics": topics,
        "translated_topics": topic_translations,
        "languages_supported": list(LANGUAGES.keys())
    }

    output_path = "translations.json"
    with open(output_path, "w",
              encoding="utf-8") as f:
        json.dump(output, f, indent=4,
                  ensure_ascii=False)


    print("="*50)
    print("TRANSLATION COMPLETE")
    print()
    print("ORIGINAL ENGLISH:")
    print("-"*50)
    print(short_summary[:300])
    print()

    for lang in LANGUAGES.keys():
        print(f"{lang.upper()} TRANSLATION:")
        print("-"*50)
        translation = translations.get(lang, "")
        print(translation[:300])
        print()

    print(f"All translations saved to: {output_path}")
    print("="*50)

    return output


if __name__ == "__main__":
    run_translation()

def translate_to_ibibio(text):
    """
    Translates to Ibibio using fine-tuned model.
    """
    from transformers import (
        AutoTokenizer,
        AutoModelForSeq2SeqLM
    )
    import torch

    tokenizer = AutoTokenizer.from_pretrained(
        "ibibio_model"
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "ibibio_model"
    )
    model.eval()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )

    target_id = tokenizer.convert_tokens_to_ids(
        "yor_Latn"
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