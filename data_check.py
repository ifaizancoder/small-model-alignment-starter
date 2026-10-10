from itertools import islice
from datasets import load_dataset
from transformers import AutoTokenizer
from my_implementation import extract_pair_from_example, build_sft_batch
from train_sft import PairDataset, make_dataloader

# 1. Load 4 examples (streaming, so nothing big is downloaded)
data = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
pairs = [extract_pair_from_example(ex) for ex in islice(data, 4)]
prompts = [p for p, r in pairs]
responses = [r for p, r in pairs]

# 2. Tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
print("eos:", tokenizer.eos_token, "| pad:", tokenizer.pad_token)

# 3. Batch builder
batch = build_sft_batch(tokenizer, prompts, responses, max_length=512)
print("batch shape:", batch["input_ids"].shape)

# What the model reads (first row, padding removed)
real_tokens = batch["attention_mask"][0] == 1
print("=== MODEL READS ===")
print(tokenizer.decode(batch["input_ids"][0][real_tokens]))

# What the model is graded on (first row)
graded = batch["labels"][0]
print("=== GRADED PART ===")
print(tokenizer.decode(graded[graded != -100]))

# 4. Dataset
dataset = PairDataset(pairs)
print("dataset size:", len(dataset))            # should print 4
print("first response:", dataset[0][1][:80])

# 5. DataLoader
loader = make_dataloader(pairs, batch_size=2, shuffle=False)
for batch_prompts, batch_responses in loader:
    print(len(batch_prompts), len(batch_responses), "|", batch_responses[0][:50])