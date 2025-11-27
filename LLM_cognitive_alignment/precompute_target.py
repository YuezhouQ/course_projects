import copy
import functools
import json
import logging
import os
import pickle
import random
import sys
import time
from typing import Optional
import warnings

import openai  # Import OpenAI's library

# Constants for OpenAI API
OPENAI_API_KEY = "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"  # Replace with your actual API key
openai.api_key = OPENAI_API_KEY

SUPPORTED_MODEL_TYPES = [
    "gpt-4o",
]

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
    model_type: str,
    output_shard_size: int = 500,
    resume: bool = False,
    compress: bool = False,
) -> None:
    """Generates text samples based on OpenAI's GPT-4 API."""

    assert model_type in SUPPORTED_MODEL_TYPES, f"Unsupported model type: {model_type}"

    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load the prompts
    with open(prompts_json_path, "r") as fp:
        prompts = json.load(fp)

    prompts = list(sorted(prompts.items(), key=lambda t: t[0]))
    prompts = prompts[0:2]

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

        if not prompt.strip():
            print(f'Skipping "{key}" (empty prompt)...')
            continue

        # Use OpenAI API to get the model response
        try:
            response = openai.ChatCompletion.create(
                model=model_type,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=0,  # No additional text generation
                logprobs=1000,  # Retrieve top-100 probabilities
                echo=True,
            )

            # Extract log probabilities and tokens
            choices = response["choices"][0]["logprobs"]
            tokens = choices["tokens"]
            top_logprobs = choices["top_logprobs"]

            outputs[key] = {
                "tokens": tokens,
                "top_logprobs": top_logprobs,
            }
        except Exception as e:
            print(f"Error processing prompt {key}: {e}")
            continue

    if len(outputs):
        write_shard(outputs, shard_path, compress=compress)

if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)
