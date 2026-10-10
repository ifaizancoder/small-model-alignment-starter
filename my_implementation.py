import torch


def encode_pair(tokenizer, prompt, response, max_length):
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    response_ids = tokenizer.encode(response, add_special_tokens=False)
    if tokenizer.eos_token_id is not None:
        response_ids = response_ids + [tokenizer.eos_token_id]
    ids = (prompt_ids + response_ids)[:max_length]
    labels = ([-100] * len(prompt_ids) + response_ids)[:max_length]
    return ids, labels


def build_sft_batch(tokenizer, prompts, responses, max_length):
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    input_ids = []
    attention_mask = []
    labels = []
    for prompt, response in zip(prompts, responses):
        ids, label = encode_pair(tokenizer, prompt, response, max_length)
        input_ids.append(ids)
        attention_mask.append([1] * len(ids))
        labels.append(label)
    max_len = max(len(ids) for ids in input_ids)
    input_ids = [ids + [pad_id] * (max_len - len(ids)) for ids in input_ids]
    attention_mask = [mask + [0] * (max_len - len(mask)) for mask in attention_mask]
    labels = [label + [-100] * (max_len - len(label)) for label in labels]
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }