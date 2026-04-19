#!/usr/bin/env python3
"""
Query embeddings from either an Ollama model or a GGUF file (via llama.cpp).

Usage:
    # Query via Ollama API
    python src/query_embeddings.py --model fin-master --input "US0378331005"

    # Query from file
    python src/query_embeddings.py --model fin-master --file data/sample_isins.txt

    # Compare two models
    python src/query_embeddings.py --model fin-master --model2 fin-master-extended --file data/sample_isins.txt

    # Query GGUF directly via llama.cpp (requires --gguf and --llama-cpp-dir)
    python src/query_embeddings.py --gguf output/model.gguf --llama-cpp-dir ./llama.cpp --input "test"
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Ollama Query
# ──────────────────────────────────────────────────────────────────────────────

def query_ollama(model: str, input: str | list[str], ollama_url: str = "http://localhost:11434") -> list[list[float]]:
    """Query embeddings from Ollama API."""
    import requests

    payload = {"model": model, "input": input}
    resp = requests.post(f"{ollama_url}/api/embed", json=payload)

    if resp.status_code != 200:
        raise RuntimeError(f"Ollama API error: {resp.status_code} {resp.text}")

    return resp.json()["embeddings"]


# ──────────────────────────────────────────────────────────────────────────────
# GGUF Direct Query (via llama.cpp server)
# ──────────────────────────────────────────────────────────────────────────────

def query_gguf_direct(gguf_path: str, input: str | list[str], llama_cpp_dir: str) -> list[list[float]]:
    """
    Query embeddings directly from a GGUF file using llama.cpp's embedding server.
    This starts a temporary server, queries it, then stops it.
    """
    import socket
    import time
    import requests

    server_path = Path(llama_cpp_dir) / "build" / "bin" / "llama-server"
    if not server_path.exists():
        # Try alternative location
        server_path = Path(llama_cpp_dir) / "build" / "bin" / "Release" / "llama-server"

    if not server_path.exists():
        raise FileNotFoundError(
            f"llama-server not found. Build llama.cpp first:\n"
            f"  cd {llama_cpp_dir} && cmake -B build && cmake --build build"
        )

    # Find a free port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('localhost', 0))
        port = s.getsockname()[1]

    cmd = [
        str(server_path),
        "--model", gguf_path,
        "--port", str(port),
        "--embedding",
        "--host", "127.0.0.1",
        "--n-gpu-layers", "0",  # CPU only
    ]

    print(f"Starting llama.cpp server on port {port}...")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Wait for server to start
    time.sleep(3)

    try:
        if isinstance(input, str):
            input = [input]

        resp = requests.post(
            f"http://127.0.0.1:{port}/embedding",
            json={"content": "\n".join(input)}
        )

        if resp.status_code != 200:
            raise RuntimeError(f"llama.cpp server error: {resp.status_code} {resp.text}")

        return resp.json()["embedding"]
    finally:
        proc.terminate()
        proc.wait()
        print("llama.cpp server stopped.")


# ──────────────────────────────────────────────────────────────────────────────
# Display and Analysis
# ──────────────────────────────────────────────────────────────────────────────

def display_embeddings(texts: list[str], embeddings: list[list[float]], prefix: str = "Model"):
    """Display embeddings in a readable format."""
    print(f"\n{'Text':<35} | {'Dims':<5} | {'L2 Norm':<10} | {'First 5 Values'}")
    print("-" * 90)

    for text, emb in zip(texts, embeddings):
        emb_np = np.array(emb, dtype=np.float32)
        norm = np.linalg.norm(emb_np)
        first5 = emb_np[:5]
        display_text = text[:32] + "..." if len(text) > 35 else text
        print(f"{display_text:<35} | {len(emb):<5} | {norm:<10.6f} | {first5}")


def compare_embeddings(
    texts: list[str],
    emb1: list[list[float]],
    emb2: list[list[float]],
    name1: str = "Model1",
    name2: str = "Model2",
):
    """Compare two sets of embeddings."""
    from sentence_transformers import util

    print(f"\n{'Text':<35} | {'Cosine Sim':<12} | {'L2 Diff':<12}")
    print("-" * 70)

    for text, e1, e2 in zip(texts, emb1, emb2):
        e1_np = np.array(e1, dtype=np.float32)
        e2_np = np.array(e2, dtype=np.float32)

        cos_sim = float(util.cos_sim(e1_np, e2_np).item())
        l2_diff = float(np.linalg.norm(e1_np - e2_np))

        display_text = text[:32] + "..." if len(text) > 35 else text
        print(f"{display_text:<35} | {cos_sim:<12.6f} | {l2_diff:<12.6f}")


def similarity_matrix(texts: list[str], embeddings: list[list[float]], model_name: str = "Model"):
    """Show pairwise cosine similarity between embeddings."""
    from sentence_transformers import util

    emb_np = np.array(embeddings, dtype=np.float32)
    sim_matrix = util.cos_sim(emb_np, emb_np)

    print(f"\n{'Pairwise Cosine Similarity (' + model_name + ')':^50}")
    print("-" * 50)
    for i, t1 in enumerate(texts):
        for j, t2 in enumerate(texts):
            if i < j:
                sim = float(sim_matrix[i][j])
                print(f"  {t1[:25]:<25} vs {t2[:25]:<25} = {sim:.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Query embeddings from Ollama or GGUF file"
    )
    parser.add_argument(
        "--model", help="Ollama model name"
    )
    parser.add_argument(
        "--model2", help="Second Ollama model name (for comparison)"
    )
    parser.add_argument(
        "--gguf", help="Path to GGUF file (alternative to --model)"
    )
    parser.add_argument(
        "--llama-cpp-dir", default="./llama.cpp",
        help="Path to llama.cpp directory (required for --gguf)"
    )
    parser.add_argument(
        "--input", help="Single text to embed"
    )
    parser.add_argument(
        "--file", help="File with texts to embed (one per line)"
    )
    parser.add_argument(
        "--ollama-url", default="http://localhost:11434",
        help="Ollama API URL"
    )
    parser.add_argument(
        "--truncate", action="store_true",
        help="Truncate long inputs to model's context length"
    )
    parser.add_argument(
        "--dimensions", type=int,
        help="Truncate output to N dimensions"
    )
    parser.add_argument(
        "--show-similarity", action="store_true",
        help="Show pairwise cosine similarity matrix"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output raw JSON instead of formatted table"
    )

    args = parser.parse_args()

    # Resolve inputs
    if not args.input and not args.file:
        print("ERROR: Provide --input or --file")
        sys.exit(1)

    if not args.model and not args.gguf:
        print("ERROR: Provide --model or --gguf")
        sys.exit(1)

    # Load input texts
    if args.file:
        texts = []
        with open(args.file, 'r', encoding='utf-8') as f:
            for line in f:
                text = line.strip()
                if text and not text.startswith('#'):
                    texts.append(text)
    else:
        texts = [args.input]

    # Query model 1
    if args.model:
        print(f"Querying Ollama model: {args.model}")
        payload = {"model": args.model, "input": texts}
        if args.truncate:
            payload["truncate"] = True
        if args.dimensions:
            payload["dimensions"] = args.dimensions

        import requests
        resp = requests.post(f"{args.ollama_url}/api/embed", json=payload)
        if resp.status_code != 200:
            print(f"Error: {resp.text}")
            sys.exit(1)

        emb1 = resp.json()["embeddings"]
        name1 = args.model
    else:
        print(f"Querying GGUF: {args.gguf}")
        emb1 = query_gguf_direct(args.gguf, texts, args.llama_cpp_dir)
        name1 = Path(args.gguf).stem

    # Query model 2 (optional comparison)
    emb2 = None
    name2 = None
    if args.model2:
        print(f"Querying Ollama model: {args.model2}")
        payload = {"model": args.model2, "input": texts}
        if args.truncate:
            payload["truncate"] = True
        if args.dimensions:
            payload["dimensions"] = args.dimensions

        import requests
        resp = requests.post(f"{args.ollama_url}/api/embed", json=payload)
        if resp.status_code != 200:
            print(f"Error: {resp.text}")
            sys.exit(1)

        emb2 = resp.json()["embeddings"]
        name2 = args.model2

    # Output
    if args.json:
        result = {
            "model": name1,
            "texts": texts,
            "embeddings": emb1,
        }
        if emb2:
            result["model2"] = name2
            result["embeddings2"] = emb2
        print(json.dumps(result, indent=2))
    else:
        print("=" * 90)
        display_embeddings(texts, emb1, prefix=name1)

        if emb2:
            print(f"\n--- Comparison: {name1} vs {name2} ---")
            compare_embeddings(texts, emb1, emb2, name1, name2)

        if args.show_similarity:
            similarity_matrix(texts, emb1, name1)

    # Print stats
    print(f"\nProcessed {len(texts)} texts")
    print(f"Embedding dimensions: {len(emb1[0]) if emb1 else 'N/A'}")


if __name__ == "__main__":
    main()
