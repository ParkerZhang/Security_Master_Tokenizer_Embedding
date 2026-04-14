#!/usr/bin/env python3
"""
Tests for the embedding model pipeline.

Run: .venv/bin/python tests/test_pipeline.py
"""

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

# Try to import requests - skip tests if not available
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import numpy as np
    from sentence_transformers import util
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
TEST_MODEL = os.environ.get("TEST_MODEL", "fin-master")


@unittest.skipIf(not HAS_REQUESTS, "requests not installed")
class TestOllamaEmbedding(unittest.TestCase):
    """Test Ollama embedding API."""

    def test_single_embedding(self):
        """Test single text embedding."""
        resp = requests.post(f"{OLLAMA_URL}/api/embed", json={
            "model": TEST_MODEL,
            "input": "Hello world"
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("embeddings", data)
        self.assertEqual(len(data["embeddings"]), 1)
        self.assertGreater(len(data["embeddings"][0]), 0)

    def test_batch_embedding(self):
        """Test batch text embedding."""
        texts = ["First sentence", "Second sentence", "Third sentence"]
        resp = requests.post(f"{OLLAMA_URL}/api/embed", json={
            "model": TEST_MODEL,
            "input": texts
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(len(data["embeddings"]), 3)

    def test_dimensions_truncation(self):
        """Test dimension truncation parameter."""
        resp = requests.post(f"{OLLAMA_URL}/api/embed", json={
            "model": TEST_MODEL,
            "input": "test",
            "dimensions": 128
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(len(data["embeddings"][0]), 128)

    def test_empty_input(self):
        """Test empty input returns empty embeddings."""
        resp = requests.post(f"{OLLAMA_URL}/api/embed", json={
            "model": TEST_MODEL,
            "input": ""
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["embeddings"], [])

    @unittest.skipIf(not HAS_SENTENCE_TRANSFORMERS, "sentence-transformers not installed")
    def test_embedding_normalization(self):
        """Test that embeddings are L2 normalized."""
        resp = requests.post(f"{OLLAMA_URL}/api/embed", json={
            "model": TEST_MODEL,
            "input": "test"
        })
        self.assertEqual(resp.status_code, 200)
        emb = np.array(resp.json()["embeddings"][0])
        norm = np.linalg.norm(emb)
        self.assertAlmostEqual(norm, 1.0, places=5)

    @unittest.skipIf(not HAS_SENTENCE_TRANSFORMERS, "sentence-transformers not installed")
    def test_cosine_similarity_identical(self):
        """Test identical inputs have cosine similarity of 1.0."""
        resp = requests.post(f"{OLLAMA_URL}/api/embed", json={
            "model": TEST_MODEL,
            "input": ["test", "test"]
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        e1 = np.array(data["embeddings"][0])
        e2 = np.array(data["embeddings"][1])
        sim = float(util.cos_sim(e1, e2).item())
        self.assertAlmostEqual(sim, 1.0, places=6)


@unittest.skipIf(not HAS_REQUESTS, "requests not installed")
class TestTokenRecognition(unittest.TestCase):
    """Test that extended tokens are recognized."""

    def test_isin_recognition(self):
        """Test ISIN codes are embedded without error."""
        isins = [
            "US0378331005",  # Apple
            "GB0002162385",  # BAE Systems
            "DE0005140008",  # Deutsche Bank
        ]
        for isin in isins:
            resp = requests.post(f"{OLLAMA_URL}/api/embed", json={
                "model": TEST_MODEL,
                "input": isin
            })
            self.assertEqual(resp.status_code, 200, f"Failed for ISIN: {isin}")
            data = resp.json()
            self.assertEqual(len(data["embeddings"]), 1)
            self.assertGreater(len(data["embeddings"][0]), 0)


if __name__ == "__main__":
    unittest.main()
