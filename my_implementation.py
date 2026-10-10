import torch
import torch.nn.functional as F

def format_prompt(instruction):
    return f"> Instruction:\n{instruction}\n\n> Response:\n"

def extract_pair_from_example(example):
    messages = example["messages"]
    instruction = messages[0]["content"]
    response = messages[1]["content"]
    prompt = format_prompt(instruction)
    return prompt, response

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

def response_logprobs(model, batch):
    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    logits = logits[:, :-1].float()          
    target = labels[:, 1:]                   
    mask = (target != -100)                  
    safe_target = target.masked_fill(~mask, 0)
    log_probs = F.log_softmax(logits, dim=-1)
    token_logps = log_probs.gather(dim=-1, index=safe_target.unsqueeze(-1)).squeeze(-1)
    token_logps = token_logps * mask         
    summed_log_probs = token_logps.sum(dim=-1)   
    return summed_log_probs

def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    batch = build_sft_batch(tokenizer, prompts, responses, max_length)
    summed_log_probs = response_logprobs(model, batch)
    return summed_log_probs

def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    batch = build_sft_batch(tokenizer, prompts, responses, max_length)
    log_probs = response_logprobs(model, batch)
    total_log_probs = log_probs.sum()
    num_tokens = max((batch["labels"][:, 1:] != -100).sum().item(), 1)
    loss = -total_log_probs / num_tokens
    return loss
