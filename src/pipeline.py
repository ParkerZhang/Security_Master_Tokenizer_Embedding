#!/usr/bin/env python3
"""
Pipeline to extend an embedding model with custom tokens (ISINs, CUSIPs, etc.).

Usage:
    python src/pipeline.py --tokens-file data/sample_isins.txt

This pipeline:
    1. Loads custom tokens from a text file
    2. Extends the tokenizer vocabulary
    3. Extends the embedding matrix (initialized as mean + small noise)
    4. Fixes sentence-transformers config for llama.cpp compatibility
    5. Converts to GGUF format
    6. Creates the Ollama model
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_MODEL_DIR = "/home/src/finaxis/my-fin-master-model"
DEFAULT_OUTPUT_DIR = "output/extended-model"
DEFAULT_GGUF_PATH = "output/model.gguf"
DEFAULT_OLLAMA_NAME = "extended-embedding"


# ──────────────────────────────────────────────────────────────────────────────
# Step 1: Load new tokens from file
# ──────────────────────────────────────────────────────────────────────────────

def load_tokens(tokens_file: str) -> list[str]:
    """Load tokens from a text file, one per line."""
    tokens = []
    with open(tokens_file, 'r', encoding='utf-8') as f:
        for line in f:
            token = line.strip()
            if token and not token.startswith('#'):
                tokens.append(token)
    return tokens


# ──────────────────────────────────────────────────────────────────────────────
# Step 2: Extend the tokenizer and embedding matrix
# ──────────────────────────────────────────────────────────────────────────────

def extend_tokenizer_and_model(model_dir: str, new_tokens: list[str], output_dir: str) -> str:
    """
    Add new tokens to the model's tokenizer and extend the embedding matrix.
    Returns the path to the updated model directory.
    """
    import torch
    from transformers import AutoTokenizer

    print(f"Loading tokenizer from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    original_vocab_size = len(tokenizer)
    print(f"Original vocabulary size: {original_vocab_size}")

    # Find tokens that aren't already in the vocabulary
    existing_tokens = set(tokenizer.get_vocab().keys())
    tokens_to_add = [t for t in new_tokens if t not in existing_tokens]

    print(f"New tokens to add: {len(tokens_to_add)} / {len(new_tokens)}")

    if not tokens_to_add:
        print("All tokens already exist in vocabulary. Skipping tokenizer extension.")
        shutil.copytree(model_dir, output_dir, dirs_exist_ok=True)
        return output_dir

    # Add tokens to tokenizer
    num_added = tokenizer.add_tokens(tokens_to_add)
    new_vocab_size = len(tokenizer)
    print(f"Added {num_added} tokens. New vocabulary size: {new_vocab_size}")

    # Save the extended tokenizer
    tokenizer.save_pretrained(output_dir)
    print(f"Extended tokenizer saved to: {output_dir}")

    # Load model state dict
    model_path = Path(model_dir)
    safetensors_path = model_path / "model.safetensors"
    if safetensors_path.exists():
        from safetensors.torch import load_file, save_file
        state_dict = load_file(str(safetensors_path))
        use_safetensors = True
    else:
        pt_path = model_path / "pytorch_model.bin"
        if pt_path.exists():
            state_dict = torch.load(str(pt_path), map_location='cpu', weights_only=True)
            use_safetensors = False
        else:
            raise FileNotFoundError(f"No model weights found in {model_dir}")

    # Find the token embedding layer
    embed_key = None
    for key in state_dict:
        if "embeddings.word_embeddings" in key or "token_embd" in key:
            embed_key = key
            break

    if embed_key is None:
        print("WARNING: Could not find token embedding layer.")
        print("New tokens will have random initialization only.")
    else:
        embed_weight = state_dict[embed_key]
        old_vocab_size, embed_dim = embed_weight.shape
        print(f"Extending embedding matrix from {old_vocab_size} to {new_vocab_size} (dim={embed_dim})")

        # Create new embedding matrix
        new_embed_weight = torch.randn(new_vocab_size, embed_dim) * 0.02

        # Copy old embeddings
        new_embed_weight[:old_vocab_size] = embed_weight

        # Initialize new embeddings as mean of existing + small noise
        mean_embed = embed_weight.mean(dim=0)
        noise = torch.randn(new_vocab_size - old_vocab_size, embed_dim) * 0.001
        new_embed_weight[old_vocab_size:] = mean_embed + noise

        state_dict[embed_key] = new_embed_weight

    # Save the extended model
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if use_safetensors:
        from safetensors.torch import save_file
        save_file(state_dict, str(output_path / "model.safetensors"))
    else:
        torch.save(state_dict, str(output_path / "pytorch_model.bin"))

    # Copy other model files (including subdirectories)
    for f in model_path.iterdir():
        if f.name not in ("model.safetensors", "pytorch_model.bin", "tokenizer.json", "tokenizer_config.json"):
            if f.is_file():
                shutil.copy2(f, output_path / f.name)
            elif f.is_dir():
                dst = output_path / f.name
                if not dst.exists():
                    shutil.copytree(f, dst)

    # Update vocab_size in config.json
    config_path = output_path / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as cf:
            config = json.load(cf)
        config["vocab_size"] = new_vocab_size
        with open(config_path, 'w') as cf:
            json.dump(config, cf, indent=2)
        print(f"Updated vocab_size in config.json to {new_vocab_size}")

    print(f"Extended model saved to: {output_dir}")
    return str(output_dir)


# ──────────────────────────────────────────────────────────────────────────────
# Step 3: Fix sentence-transformers config for llama.cpp compatibility
# ──────────────────────────────────────────────────────────────────────────────

def fix_sentence_transformers_config(model_dir: str):
    """Fix modules.json and 1_Pooling/config.json for llama.cpp converter."""
    model_path = Path(model_dir)

    # Fix modules.json
    modules_path = model_path / "modules.json"
    if modules_path.exists():
        with open(modules_path, 'r') as f:
            modules = json.load(f)

        replacements = {
            "sentence_transformers.base.modules.transformer.": "sentence_transformers.models.",
            "sentence_transformers.sentence_transformer.modules.pooling.": "sentence_transformers.models.",
            "sentence_transformers.sentence_transformer.modules.normalize.": "sentence_transformers.models.",
        }

        for m in modules:
            old_type = m.get("type", "")
            for old, new in replacements.items():
                old_type = old_type.replace(old, new)
            m["type"] = old_type

        with open(modules_path, 'w') as f:
            json.dump(modules, f, indent=2)

    # Fix 1_Pooling/config.json
    pooling_path = model_path / "1_Pooling" / "config.json"
    if pooling_path.exists():
        with open(pooling_path, 'r') as f:
            pooling_config = json.load(f)

        if "pooling_mode" in pooling_config:
            mode = pooling_config.pop("pooling_mode")
            pooling_config["pooling_mode_mean_tokens"] = (mode == "mean")
            pooling_config["pooling_mode_cls_token"] = (mode == "cls")
            pooling_config["pooling_mode_lasttoken"] = (mode == "lasttoken")

            with open(pooling_path, 'w') as f:
                json.dump(pooling_config, f, indent=2)
            print(f"Fixed 1_Pooling/config.json: pooling_mode -> {mode}")


# ──────────────────────────────────────────────────────────────────────────────
# Step 4: Convert to GGUF
# ──────────────────────────────────────────────────────────────────────────────

def convert_to_gguf(model_dir: str, gguf_path: str, converter_path: str):
    """Convert HuggingFace model to GGUF format using llama.cpp converter."""
    print(f"Converting to GGUF: {gguf_path}")

    cmd = [
        sys.executable, converter_path,
        model_dir,
        "--outfile", gguf_path,
        "--outtype", "f16"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Conversion failed:")
        print(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
        sys.exit(1)

    print(f"GGUF file created: {gguf_path}")
    return gguf_path


# ──────────────────────────────────────────────────────────────────────────────
# Step 5: Create Ollama model
# ──────────────────────────────────────────────────────────────────────────────

def create_ollama_model(model_name: str, gguf_path: str):
    """Create an Ollama model from the GGUF file."""
    print(f"Creating Ollama model: {model_name}")

    abs_gguf = str(Path(gguf_path).resolve())

    modelfile_path = Path(gguf_path).parent / "Modelfile"
    with open(modelfile_path, 'w') as f:
        f.write(f"FROM {abs_gguf}\n")

    cmd = ["ollama", "create", model_name, "-f", str(modelfile_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Ollama create failed: {result.stderr}")
        sys.exit(1)

    print(f"Ollama model created: {model_name}")

    # Clean up temp Modelfile
    modelfile_path.unlink()


# ──────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extend an embedding model with custom tokens and rebuild for Ollama"
    )
    parser.add_argument(
        "--tokens-file", required=True,
        help="Text file with new tokens (one per line)"
    )
    parser.add_argument(
        "--model-dir", default=None,
        help="Source model directory (default: environment variable MODEL_DIR or hardcoded default)"
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory for extended model"
    )
    parser.add_argument(
        "--gguf-path", default=None,
        help="Output GGUF file path"
    )
    parser.add_argument(
        "--ollama-name", default=None,
        help="Ollama model name"
    )
    parser.add_argument(
        "--llama-cpp-dir", default=None,
        help="Path to llama.cpp directory (default: environment variable LLAMA_CPP_DIR or ./llama.cpp)"
    )
    parser.add_argument(
        "--skip-ollama", action="store_true",
        help="Skip Ollama model creation (only build GGUF)"
    )

    args = parser.parse_args()

    # Resolve paths
    project_root = Path(__file__).parent.parent
    model_dir = args.model_dir or DEFAULT_MODEL_DIR
    output_dir = args.output_dir or DEFAULT_OUTPUT_DIR
    output_dir_path = Path(output_dir)
    if not output_dir_path.is_absolute():
        output_dir_path = project_root / output_dir
        output_dir = str(output_dir_path)

    gguf_path = args.gguf_path or DEFAULT_GGUF_PATH
    gguf_path = Path(gguf_path)
    if not gguf_path.is_absolute():
        gguf_path = project_root / gguf_path
    gguf_path = str(gguf_path)

    ollama_name = args.ollama_name or DEFAULT_OLLAMA_NAME

    llama_cpp_dir = args.llama_cpp_dir
    if not llama_cpp_dir:
        llama_cpp_dir = str(project_root / "llama.cpp")
    converter_path = str(Path(llama_cpp_dir) / "convert_hf_to_gguf.py")

    if not Path(converter_path).exists():
        print(f"ERROR: llama.cpp converter not found at {converter_path}")
        print(f"Clone llama.cpp first:")
        print(f"  git clone https://github.com/ggml-org/llama.cpp.git {llama_cpp_dir}")
        sys.exit(1)

    print("=" * 70)
    print("  EXTEND EMBEDDING MODEL WITH CUSTOM TOKENS")
    print("=" * 70)

    # Step 1: Load tokens
    print("\n[Step 1] Loading tokens from file...")
    tokens = load_tokens(args.tokens_file)
    if not tokens:
        print("ERROR: No tokens found in file.")
        sys.exit(1)
    print(f"Loaded {len(tokens)} tokens:")
    for t in tokens[:10]:
        print(f"  - {t}")
    if len(tokens) > 10:
        print(f"  ... and {len(tokens) - 10} more")

    # Step 2: Extend tokenizer and model
    print("\n[Step 2] Extending tokenizer and model...")
    extended_dir = extend_tokenizer_and_model(model_dir, tokens, output_dir)

    # Step 3: Fix configs for llama.cpp
    print("\n[Step 3] Fixing sentence-transformers configs...")
    fix_sentence_transformers_config(extended_dir)

    # Step 4: Convert to GGUF
    print("\n[Step 4] Converting to GGUF...")
    convert_to_gguf(extended_dir, gguf_path, converter_path)

    # Step 5: Create Ollama model
    if not args.skip_ollama:
        print("\n[Step 5] Creating Ollama model...")
        create_ollama_model(ollama_name, gguf_path)
    else:
        print("\n[Step 5] Skipped (--skip-ollama)")

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\nExtended model:   {extended_dir}")
    print(f"GGUF file:        {gguf_path}")
    if not args.skip_ollama:
        print(f"Ollama model:     {ollama_name}")
    print(f"\nTest with:")
    print(f"  python src/query_embeddings.py --model {ollama_name} --input \"US0378331005\"")


if __name__ == "__main__":
    main()
