import copy
import functools
import json
import logging
import os
import pickle
import sys
import time
from typing import Optional
import warnings
import random


import lightning as L
import torch
from litgpt import LLM, Config, Tokenizer  # Updated imports for litgpt

# Constants for dtype and device
DTYPE = torch.bfloat16
DEVICE = torch.device('cuda:0')
SUPPORTED_MODEL_TYPES = set([
    "meta-llama/Meta-Llama-3.1-8B",
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-3.2-1B",
    "microsoft/phi-2",
])

def write_shard(outputs, shard_path, compress=False):
    extension = shard_path.split('.')[-1]
    if not compress:
        assert extension == "pickle"
        open_fn = open
    else:
        assert extension == "xz"
        open_fn = lzma.open
   
    with open_fn(shard_path, "wb") as fp:
        pickle.dump(outputs, fp, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Wrote shard {shard_path}...")

def main(
    *,
    prompts_json_path: str,
    output_dir: str,
    checkpoint_path: str, 
    model_type: str,
    tokenizer_path: Optional[str] = None,
    output_shard_size: int = 200,
    return_embeddings: bool = False,
    return_initial_embeddings: bool = False,
    return_after_layer_n: Optional[int] = None,
    resume: bool = False,
    compress: bool = False,
) -> None:
    """Generates text samples based on a pre-trained model and tokenizer."""

    assert model_type in SUPPORTED_MODEL_TYPES, f"Unsupported model type: {model_type}"

    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load the model and tokenizer using litgpt
    model = LLM.load(f"{checkpoint_path}/{model_type}")
    tokenizer = Tokenizer(f"{checkpoint_path}/{model_type}")
    DEVICE = torch.device("mps")  # Use MPS for Apple Silicon

    model.eval()

    # Load the prompts
    with open(prompts_json_path, "r") as fp:
        prompts = json.load(fp)
        
    random.seed(123)
    prompts = list(sorted(prompts.items(), key=lambda t: t[0]))
    prompts = random.sample(prompts, 1000)

    def get_shard_path(shard_count, compress=False):
        output_basename = os.path.split(prompts_json_path)[-1].split('.')[0]
        output_basename += f"_{model_type.replace('/', '_')}"  # Replace '/' in model_type with '_'
        shard_name = f"{output_basename}_{shard_count}.pickle"
        if compress:
            shard_name += ".xz"
        shard_path = os.path.join(output_dir, shard_name)

        # Ensure the directory exists
        os.makedirs(os.path.dirname(shard_path), exist_ok=True)

        return shard_path


    # Generate logits
    if resume:
        print(f"Resuming computation...")

    shard_count = 0
    shard_path = get_shard_path(shard_count, compress=compress)
    outputs = {}
    for i, (key, prompt) in enumerate(prompts):
        # Write shard
        if i != 0 and i % output_shard_size == 0:
            if len(outputs):
                write_shard(outputs, shard_path, compress=compress)

            shard_count += 1
            shard_path = get_shard_path(shard_count, compress=compress)
            outputs = {}

        # Skip precomputed shard entries
        if resume and os.path.isfile(shard_path):
            continue

        # Tokenize the prompt
        encoded_prompt = tokenizer.encode(prompt).to(DEVICE)

        len_prompt = len(encoded_prompt)
        encoded_prompt = encoded_prompt.unsqueeze(0)  # Add batch dimension

        if len_prompt == 0:
            print(f'Skipping "{key}" (too short)...')
            continue

        max_len = model.config.block_size if hasattr(model.config, "block_size") else 512

        if len_prompt > max_len:
            logging.info(f'Truncating {key}...')
            encoded_prompt = encoded_prompt[..., :max_len]
            len_prompt = max_len

        # Run the model
        with torch.no_grad():
            logits = model(encoded_prompt)

        logits = logits.squeeze(0).cpu()
        top_k_logits, top_k_indices = torch.topk(logits, k=1000, dim=-1)
        top_k_probs = torch.softmax(top_k_logits, dim=-1)
        outputs[key] = {"probs": top_k_probs.cpu(), "indices": top_k_indices.cpu()}

    if len(outputs):
        write_shard(outputs, shard_path, compress=compress)


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    warnings.filterwarnings(
        "ignore", 
        message="ComplexHalf support is experimental and many operators don't support it yet"
    )
    warnings.filterwarnings(
        "ignore", 
        message="MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization",
    )
    CLI(main)
