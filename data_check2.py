from itertools import islice

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from my_implementation import extract_pair_from_example, compute_sft_loss

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")

data = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
pairs = [extract_pair_from_example(ex) for ex in islice(data, 2)]
prompts = [p for p, r in pairs]
responses = [r for p, r in pairs]

loss = compute_sft_loss(model, tokenizer, prompts, responses, max_length=512)
print("loss:", loss.item())

loss.backward()
print("gradients flow:", any(p.grad is not None for p in model.parameters()))