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

def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length):
    chosen_batch = build_sft_batch(tokenizer, prompts, chosen_responses, max_length)
    rejected_batch = build_sft_batch(tokenizer, prompts, rejected_responses, max_length)
    return {"chosen": chosen_batch, "rejected": rejected_batch}

def compute_dpo_loss_from_logps(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps,reference_rejected_logps,beta):
    policy_diff = policy_chosen_logps - policy_rejected_logps
    reference_diff = reference_chosen_logps - reference_rejected_logps
    loss = -F.logsigmoid(beta * (policy_diff - reference_diff))
    return loss.mean()

def compute_dpo_loss(policy_model, reference_model, tokenizer, prompts, chosen_responses, rejected_responses, beta, max_length):
    policy_chosen_logps = compute_response_logprobs(policy_model, tokenizer, prompts, chosen_responses, max_length)
    policy_rejected_logps = compute_response_logprobs(policy_model, tokenizer, prompts, rejected_responses, max_length)
    with torch.no_grad():
        reference_chosen_logps = compute_response_logprobs(reference_model, tokenizer, prompts, chosen_responses, max_length)
        reference_rejected_logps = compute_response_logprobs(reference_model, tokenizer, prompts, rejected_responses, max_length)
    reference_chosen_logps = reference_chosen_logps.to(policy_chosen_logps.device)
    reference_rejected_logps = reference_rejected_logps.to(policy_chosen_logps.device)
    loss = compute_dpo_loss_from_logps(policy_chosen_logps, policy_rejected_logps,reference_chosen_logps, reference_rejected_logps,beta)
    return loss

def split_hh_example(example):
    chosen, rejected = example["chosen"], example["rejected"]
    human, assistant = "\n\nHuman:", "\n\nAssistant:"
    for text in (chosen, rejected):
        if text.count(human) != 1 or text.count(assistant) != 1:
            return None
    chosen_prompt, chosen_response = chosen.split(assistant)
    rejected_prompt, rejected_response = rejected.split(assistant)
    if chosen_prompt != rejected_prompt:
        return None
    instruction = chosen_prompt.replace(human, "").strip()
    chosen_response = chosen_response.strip()
    rejected_response = rejected_response.strip()
    if not instruction or not chosen_response or not rejected_response:
        return None
    if chosen_response == rejected_response:      
        return None
    return format_prompt(instruction), chosen_response, rejected_response

def split_uf_example(example):
    chosen, rejected = example["chosen"], example["rejected"]
    if len(chosen) != 2 or len(rejected) != 2:
        return None
    if example["score_chosen"] <= example["score_rejected"]:
        return None
    instruction = example["prompt"].strip()
    chosen_response = chosen[1]["content"].strip()
    rejected_response = rejected[1]["content"].strip()
    if not instruction or not chosen_response or not rejected_response:
        return None
    if chosen_response == rejected_response:
        return None
    return format_prompt(instruction), chosen_response, rejected_response

