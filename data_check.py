from datasets import load_dataset
from itertools import islice
from transformers import AutoTokenizer
from my_implementation import extract_pair_from_example, build_sft_batch
data = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
example = next(iter(data))  

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
print("eos:", tokenizer.eos_token, "| pad:", tokenizer.pad_token)

pairs = [extract_pair_from_example(ex) for ex in islice(data, 4)]
prompts = [p for p, r in pairs]
responses = [r for p, r in pairs]

batch = build_sft_batch(tokenizer, prompts, responses, max_length=512)
print(batch["input_ids"].shape)

# What the model reads (first row)
print(tokenizer.decode(batch["input_ids"][0]))

# What the model is graded on (first row)
graded = batch["labels"][0]
print("=== GRADED PART ===")
print(tokenizer.decode(graded[graded != -100]))