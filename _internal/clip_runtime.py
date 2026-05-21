"""
OpenCLIP model runtime — loads once, reused across calls.

Exposes two operations:
  - embed_images(pil_images)   -> list of raw 512-d numpy arrays  (for embedding.py)
  - score_images(pil_images)   -> list of (label, p_class0, probs) (for clustering.py)
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

# Default building-positive / negative prompts (can be overridden at runtime init).
DEFAULT_TEXT_PROMPTS: Tuple[str, ...] = (
    "a photograph taken outdoors showing the real exterior facade of a built structure "
    "with walls windows or roof visible in the scene",
    "a close-up photograph of paper pages documents blueprints floor plans maps letters "
    "newspapers books or archival sheets with printed text or diagrams",
    "a photograph of a computer screen tablet projector slide framed poster or museum label "
    "displaying an image or drawing of a building",
)


class ClipRuntime:
    """
    Holds the loaded OpenCLIP model, preprocessor, and pre-encoded text features.
    Instantiate once and reuse across batches.

    Parameters
    ----------
    model_name   : OpenCLIP architecture, e.g. "ViT-B-32"
    pretrained   : checkpoint tag, e.g. "laion2b_s34b_b79k"
    device       : "cuda", "cpu", or None for auto-detect
    text_prompts : custom (class0_positive, *negatives) prompt tuple; falls back to DEFAULT_TEXT_PROMPTS
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        *,
        device: Optional[str] = None,
        text_prompts: Optional[Sequence[str]] = None,
    ) -> None:
        import torch
        import open_clip

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(self.device).eval()
        self._tokenizer = open_clip.get_tokenizer(model_name)

        prompts = tuple(text_prompts) if text_prompts else DEFAULT_TEXT_PROMPTS
        if len(prompts) < 2:
            raise ValueError("At least 2 prompts required (class 0 = positive, rest = negatives).")
        self.text_prompts = prompts

        tokens = self._tokenizer(list(self.text_prompts)).to(self.device)
        with torch.no_grad():
            tf = self.model.encode_text(tokens)
            self.text_features = tf / tf.norm(dim=-1, keepdim=True)
        self._logit_scale = self.model.logit_scale.exp()

        self.embed_dim: int = self.model.visual.output_dim  # 512 for ViT-B-32

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed_images(self, images: List[object]) -> List[np.ndarray]:
        """
        Compute L2-normalised image embeddings.

        Returns a list of 1-D float32 numpy arrays of length ``embed_dim`` (512 for ViT-B-32).
        Failed / None images must be filtered out by the caller before passing here.
        """
        import torch

        if not images:
            return []
        tensors = torch.stack([self.preprocess(im) for im in images]).to(self.device)
        with torch.no_grad():
            feat = self.model.encode_image(tensors)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return [arr.astype(np.float32) for arr in feat.cpu().numpy()]

    # ------------------------------------------------------------------
    # Vision scoring (softmax over text prompts)
    # ------------------------------------------------------------------

    def score_images(
        self, images: List[object]
    ) -> List[Tuple[bool, float, Tuple[float, ...]]]:
        """
        Classify each image against ``self.text_prompts`` via softmax.

        Returns a list of (is_building, p_building, full_probs_tuple) per image.
        is_building is True when argmax == 0 (building-positive prompt), False otherwise.
        p_building is the softmax probability for class 0.
        """
        import torch

        if not images:
            return []
        tensors = torch.stack([self.preprocess(im) for im in images]).to(self.device)
        with torch.no_grad():
            feat = self.model.encode_image(tensors)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            logits = self._logit_scale * (feat @ self.text_features.T)
            probs = logits.softmax(dim=-1).cpu().numpy()

        results = []
        for row in probs:
            p0 = float(row[0])
            pred = int(row.argmax())
            results.append((pred == 0, p0, tuple(float(x) for x in row)))
        return results
