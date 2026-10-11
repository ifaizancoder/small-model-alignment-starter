import sys
import train_dpo
from my_implementation import split_uf_example
 
 
def load_uf_triples(tokenizer, n_train, n_val, seed, max_prompt_tokens):
    """Seeded sample of single-turn (prompt, chosen, rejected) triples from UltraFeedback."""
    from datasets import load_dataset
 
    data = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs").shuffle(seed=seed)
    triples = []
    for example in data:
        triple = split_uf_example(example)
        if triple is None:
            continue
        prompt, chosen, rejected = triple
        # skip long prompts, same rule as SFT and the HH-RLHF run
        if len(tokenizer.encode(prompt, add_special_tokens=False)) > max_prompt_tokens:
            continue
        triples.append(triple)
        if len(triples) >= n_train + n_val:
            break
    if len(triples) < n_train + n_val:
        print(f"Warning: only found {len(triples)} valid examples, fewer than requested {n_train + n_val}")
    return triples[:n_train], triples[n_train:n_train + n_val]
 
 
if __name__ == "__main__":
    train_dpo.load_hh_triples = load_uf_triples
    if not any(arg.startswith("--output_dir") for arg in sys.argv[1:]):
        sys.argv += ["--output_dir", "outputs/dpo_uf"]
    print("dataset: UltraFeedback (HuggingFaceH4/ultrafeedback_binarized, train_prefs)")
    train_dpo.main()