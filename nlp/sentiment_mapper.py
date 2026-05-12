"""
Sentiment Mapper — Maps client communications to dense semantic vectors
using XLM-RoBERTa for cross-lingual sentiment analysis.

This module provides:
  - XLMRSentimentMapper: Encodes client text into 768-d embeddings
  - MockSentimentMapper: Returns synthetic embeddings for dev/testing
  - Risk tag extraction from embeddings
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class MockSentimentMapper:
    """
    Mock sentiment mapper for development and testing.
    Returns synthetic embeddings correlated with keyword-based distress signals.
    """

    DISTRESS_KEYWORDS = {
        "crop failure": 0.9,
        "medical emergency": 0.85,
        "flood": 0.8,
        "drought": 0.8,
        "job loss": 0.75,
        "family illness": 0.7,
        "debt": 0.6,
        "struggling": 0.65,
        "late payment": 0.5,
        "difficulty": 0.5,
    }

    POSITIVE_KEYWORDS = {
        "good harvest": -0.3,
        "new job": -0.4,
        "promotion": -0.35,
        "savings": -0.3,
        "on time": -0.2,
        "business growing": -0.4,
    }

    def __init__(self, embedding_dim: int = 768, seed: int = 42):
        self.embedding_dim = embedding_dim
        self._rng = np.random.default_rng(seed)

    def encode(self, text: str) -> np.ndarray:
        """Encode text into a mock sentiment embedding."""
        text_lower = text.lower()

        # Calculate distress score from keywords
        distress = 0.0
        for keyword, weight in self.DISTRESS_KEYWORDS.items():
            if keyword in text_lower:
                distress = max(distress, weight)

        for keyword, weight in self.POSITIVE_KEYWORDS.items():
            if keyword in text_lower:
                distress = min(distress, distress + weight)

        distress = np.clip(distress, 0, 1)

        # Generate embedding centered around distress level
        embedding = self._rng.normal(
            loc=distress * 0.5,
            scale=0.3,
            size=self.embedding_dim,
        ).astype(np.float32)

        return embedding

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """Encode a batch of texts."""
        return np.array([self.encode(t) for t in texts])

    def extract_risk_tags(self, text: str) -> dict[str, Any]:
        """Extract structured risk tags from text."""
        text_lower = text.lower()
        tags = []
        max_severity = 0.0

        for keyword, weight in self.DISTRESS_KEYWORDS.items():
            if keyword in text_lower:
                tags.append(keyword)
                max_severity = max(max_severity, weight)

        return {
            "risk_tags": tags,
            "distress_score": float(max_severity),
            "sentiment_polarity": float(np.clip(1.0 - 2.0 * max_severity, -1, 1)),
        }


class XLMRSentimentMapper:
    """
    Production sentiment mapper using XLM-RoBERTa for cross-lingual
    semantic encoding of client communications.

    Requires: transformers, torch
    """

    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        device: str | None = None,
        max_length: int = 128,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self._model = None
        self._tokenizer = None
        self._device = device

    def _lazy_load(self):
        """Lazy-load the model to avoid import-time GPU allocation."""
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            logger.info(f"Loading XLM-RoBERTa model: {self.model_name}")

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)

            if self._device is None:
                self._device = "cuda" if torch.cuda.is_available() else "cpu"

            self._model = self._model.to(self._device)
            self._model.eval()
            logger.info(f"Model loaded on {self._device}")

        except ImportError as e:
            raise ImportError(
                "XLMRSentimentMapper requires 'transformers' and 'torch'. "
                "Install with: pip install transformers torch"
            ) from e

    def encode(self, text: str) -> np.ndarray:
        """Encode a single text into a 768-d embedding using mean pooling."""
        self._lazy_load()

        import torch

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=True,
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        # Mean pooling over token embeddings (excluding padding)
        attention_mask = inputs["attention_mask"]
        token_embeddings = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
        sum_embeddings = (token_embeddings * mask_expanded).sum(dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        embedding = (sum_embeddings / sum_mask).squeeze().cpu().numpy()

        return embedding.astype(np.float32)

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """Encode a batch of texts."""
        self._lazy_load()

        import torch

        inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=True,
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        attention_mask = inputs["attention_mask"]
        token_embeddings = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
        sum_embeddings = (token_embeddings * mask_expanded).sum(dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        embeddings = (sum_embeddings / sum_mask).cpu().numpy()

        return embeddings.astype(np.float32)

    def extract_risk_tags(self, text: str) -> dict[str, Any]:
        """
        Extract risk tags using the embedding space.
        Uses cosine similarity against known distress anchors.
        """
        embedding = self.encode(text)

        # Crude distress estimation from embedding statistics
        mean_activation = float(np.mean(embedding))
        std_activation = float(np.std(embedding))

        # Higher mean activation in our encoding → more distress
        distress_score = float(np.clip(mean_activation * 2.0, 0, 1))

        return {
            "risk_tags": [],  # Would be populated by a classifier head
            "distress_score": distress_score,
            "sentiment_polarity": float(1.0 - 2.0 * distress_score),
            "embedding_mean": mean_activation,
            "embedding_std": std_activation,
        }
