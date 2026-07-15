"""MUSE (Multilingual Universal Sentence Encoder) language embeddings.

Ported from steering-with-failures' `steering/networks/language.py`: the
critic's (and DSRL noise-actor's) language conditioning uses 512-d MUSE
embeddings of the task-description string, rather than one-hot task ids.
The TF-Hub model is frozen and CPU-only (GPU VRAM is reserved for JAX);
embeddings are plain numpy arrays so nothing pulls TF into the JAX graph.

Requires `tensorflow`, `tensorflow-hub`, and `tensorflow-text` (see
requirements.txt). The first call downloads the model (~250 MB) into the
TF-Hub cache.
"""

from __future__ import annotations

from typing import List, Union

import numpy as np

MUSE_URL = "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
EMBED_DIM = 512


class MUSEEncoder:
    """Thin wrapper around the frozen TF-Hub MUSE model.

    Returns plain numpy float32 arrays so nothing pulls TF into the JAX graph.
    """

    def __init__(self, hub_url: str = MUSE_URL) -> None:
        self._model = None
        self._hub_url = hub_url

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import tensorflow as tf  # type: ignore
            import tensorflow_hub as hub  # type: ignore
            import tensorflow_text  # noqa: F401 -- registers SentencepieceOp
        except ImportError as exc:
            raise ImportError(
                "MUSEEncoder needs tensorflow, tensorflow-hub, and "
                "tensorflow-text (see requirements.txt)."
            ) from exc
        # Keep TF off the GPU -- VRAM is reserved for JAX and the render workers.
        tf.config.set_visible_devices([], "GPU")
        self._model = hub.load(self._hub_url)

    @property
    def embed_dim(self) -> int:
        return EMBED_DIM

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Encode instruction(s) to (N, 512) float32."""
        self._load()
        if isinstance(texts, str):
            texts = [texts]
        return np.asarray(self._model(texts).numpy(), dtype=np.float32)
