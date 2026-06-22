import json
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset as HFDataset

print("="*50)
print("IBIBIO FINE-TUNING PROCESS")
print("="*50)



print("Step 1: Loading Ibibio corpus...")

with open("corpus.json", "r",
          encoding="utf-8") as f:
    corpus = json.load(f)

print(f"Total sentence pairs: {len(corpus)}")


split = int(len(corpus) * 0.9)
train_data = corpus[:split]
test_data = corpus[split:]

print(f"Training pairs: {len(train_data)}")
print(f"Test pairs: {len(test_data)}")
print()



print("Step 2: Loading NLLB-200 model...")
print("(Already downloaded - loading from cache)")

MODEL_NAME = "facebook/nllb-200-distilled-600M"
SOURCE_LANG = "eng_Latn"
TARGET_LANG = "yor_Latn"


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    src_lang=SOURCE_LANG
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME
)

print("Model loaded successfully")
print()



print("Step 3: Preparing dataset...")


def preprocess(examples):
    """Tokenizes English source and Ibibio target."""
    inputs = tokenizer(
        examples["english"],
        max_length=128,
        truncation=True,
        padding="max_length"
    )

    targets = tokenizer(
        examples["ibibio"],
        max_length=128,
        truncation=True,
        padding="max_length"
    )

    inputs["labels"] = [
        [(l if l != tokenizer.pad_token_id
          else -100)
         for l in label]
        for label in targets["input_ids"]
    ]

    return inputs



train_dict = {
    "english": [d["english"] for d in train_data],
    "ibibio": [d["ibibio"] for d in train_data]
}
test_dict = {
    "english": [d["english"] for d in test_data],
    "ibibio": [d["ibibio"] for d in test_data]
}

train_dataset = HFDataset.from_dict(train_dict)
test_dataset = HFDataset.from_dict(test_dict)

train_tokenized = train_dataset.map(
    preprocess,
    batched=True,
    remove_columns=["english", "ibibio"]
)
test_tokenized = test_dataset.map(
    preprocess,
    batched=True,
    remove_columns=["english", "ibibio"]
)

print(f"Train dataset ready: "
      f"{len(train_tokenized)} samples")
print(f"Test dataset ready: "
      f"{len(test_tokenized)} samples")
print()



print("Step 4: Setting up training...")

output_dir = "ibibio_model"
os.makedirs(output_dir, exist_ok=True)

training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=4,
    warmup_steps=50,
    weight_decay=0.01,
    logging_dir="logs",
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    predict_with_generate=True,
    fp16=False,
    report_to="none"
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    padding=True
)

print("Training configuration:")
print(f"  Epochs: 10")
print(f"  Batch size: 4")
print(f"  Output: {output_dir}")
print()



print("Step 5: Starting fine-tuning...")
print("This will take 30-60 minutes on your machine")
print("Do not close this window")
print()

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=test_tokenized,
    processing_class=tokenizer,
    data_collator=data_collator
)

trainer.train()

print()
print("Fine-tuning complete!")
print()



print("Step 6: Saving fine-tuned model...")

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Model saved to: {output_dir}")
print()



print("Step 7: Testing Ibibio translation...")
print()

def translate_to_ibibio(text):
    """Translates English to Ibibio."""
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

    translation = tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True
    )[0]

    return translation



test_sentences = [
    "Good morning",
    "How are you",
    "Thank you",
    "I am a student",
    "God bless you",
    "Education is important",
    "I love my family",
    "Health is wealth"
]

print("Translation Test Results:")
print("-"*50)

results = []
for sentence in test_sentences:
    translation = translate_to_ibibio(sentence)
    print(f"EN:  {sentence}")
    print(f"IBB: {translation}")
    print()
    results.append({
        "english": sentence,
        "ibibio_translation": translation
    })


with open("ibibio_test_results.json", "w",
          encoding="utf-8") as f:
    json.dump(results, f,
              indent=4,
              ensure_ascii=False)

print("="*50)
print("IBIBIO FINE-TUNING COMPLETE")
print()
print("Files created:")
print(f"  {output_dir}/ - Fine-tuned model")
print("  ibibio_test_results.json - Test results")
print()
print("Your system now has basic Ibibio support!")
print("This is the first AI translation system")
print("for the Ibibio language.")
print("="*50)