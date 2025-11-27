import copy
import json
import logging
import os
import pickle
import random
import sys
import time
from typing import Optional
import warnings

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
    precomputed_small_emb_dir: str,
    precomputed_large_emb_dir: str,
    output_dir: str,
    entropy_min: float = 2.0,
    entropy_max: float = -1,
    no_bins: int = 2,
    min_bin: float = 0.0,
    max_bin: float = 1.0,
    balanced_classes: bool = True,
    seed: int = 42,
) -> None:
    """
    Create a gap dataset using precomputed embeddings and top-k probabilities.

    Args:
        precomputed_small_emb_dir: Directory containing small model embeddings.
        precomputed_large_emb_dir: Directory containing large model embeddings.
        output_dir: Where to save the output dataset.
        entropy_min: Minimum entropy for small model filtering.
        entropy_max: Maximum entropy (use -1 for no upper bound).
        no_bins: Number of bins for large model entropy.
        min_bin: Minimum bin value for large model entropy discretization.
        max_bin: Maximum bin value for large model entropy discretization.
        balanced_classes: Whether to balance the classes by sampling.
    """
    args = locals()
    torch.manual_seed(seed)

    if entropy_max == -1:
        entropy_max = float("inf")

    # Make output directories
    os.makedirs(output_dir, exist_ok=True)
    for subdir in ["filter", "large_entropy", "small_entropy"]:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

    # Load precomputed top-k probabilities
    def load_shard_data(directory):
        data = {}
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if file.endswith(".pickle"):
                with open(file_path, "rb") as fp:
                    data.update(pickle.load(fp))
        return data

    print("Loading precomputed embeddings...")
    small_data = load_shard_data(precomputed_small_emb_dir)
    large_data = load_shard_data(precomputed_large_emb_dir)

    filt = {}
    by_label = {}

    print("Computing entropy and creating dataset...")
    for key in small_data:
        small_probs = small_data[key]["probs"]
        large_probs = large_data[key]["probs"]

        # Compute entropy for small and large model
        small_entropy = -torch.sum(small_probs * torch.log(small_probs + 1e-8), dim=-1)
        large_entropy = -torch.sum(large_probs * torch.log(large_probs + 1e-8), dim=-1)

        # Filter small model entropy
        small_entropy_in_range = torch.logical_and(
            small_entropy >= entropy_min, 
            small_entropy < entropy_max
        )

        # Bin large model entropy
        large_entropy_bins = torch.bucketize(
            large_entropy, torch.linspace(min_bin, max_bin, steps=no_bins + 1)
        )

        for i in range(no_bins):
            by_label.setdefault(str(i), {})
            by_label[str(i)][key] = torch.logical_and(
                small_entropy_in_range,
                large_entropy_bins == i
            )

    # Balance classes
    if balanced_classes:
        print("Balancing classes...")
        sizes = {
            label: sum([torch.sum(v).item() for v in by_label[label].values()]) 
            for label in by_label
        }

        min_size = min(sizes.values())
        for label in by_label:
            for key in by_label[label]:
                mask = by_label[label][key]
                sampled_mask = mask & (torch.rand_like(mask, dtype=torch.float32) < min_size / sizes[label])
                by_label[label][key] = sampled_mask

    # Combine filters
    filt = {
        key: torch.stack([by_label[label][key] for label in by_label], dim=-1).any(dim=-1).to(torch.bool)
        for key in by_label[list(by_label.keys())[0]]
    }

    # Save filter
    output_path = os.path.join(output_dir, "filter.pickle")
    with open(output_path, "wb") as fp:
        pickle.dump(filt, fp, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Filter saved to {output_path}")

    # Save arguments
    with open(os.path.join(output_dir, "args.json"), "w") as fp:
        json.dump(args, fp)

if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)
