from datasets import load_dataset
from itertools import islice
from my_implementation import split_hh_example

data = load_dataset("Anthropic/hh-rlhf", split="train", streaming=True)
kept = [t for t in (split_hh_example(ex) for ex in islice(data, 2000)) if t is not None]
print("kept", len(kept), "of 2000")

prompt, chosen, rejected = kept[0]
print(prompt)
print("CHOSEN  :", chosen[:600])
print("REJECTED:", rejected[:600])