"""History-slot configuration shared by MIKASA training data and evaluation.

Kept free of TensorFlow/TFDS imports so evaluation can build identical history inputs.

    slots = [first frame (optional)] + [t - stride*K, ..., t - stride]   (oldest -> newest)

View order (``history_obs_views="both"``):
    default:                    [base_t, wrist_t, base_s1, wrist_s1, ..., base_sN, wrist_sN]
    history_time_embedding:     [base_s1, wrist_s1, ..., base_sN, wrist_sN, base_t, wrist_t]
With ``history_time_embedding`` the views are in time order (oldest -> current) and the model adds a
shared camera embedding (base / wrist) plus a learned per-time-position embedding instead of one
embedding per view position.

With ``history_tokens`` the views keep the default order, but only the 2 current views go through the
original TurboVLA path (V-L interaction); the history views go through DINOv3 without gradients, are
average-pooled to ``history_pool`` x ``history_pool`` tokens each, detached, mapped by a separate memory
projector, and appended to the action decoder memory after the state tokens.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image

HISTORY_OBS_VIEWS = ("both", "base")


def resize_camera_image(image: np.ndarray, size: int) -> np.ndarray:
    """Resize a uint8 HxWx3 camera image to size x size (Lanczos, like OpenVLA-OFT's MIKASA pipeline).

    Shared by training and evaluation so both produce identical pixels.
    """
    if image.shape[0] == size and image.shape[1] == size:
        return image
    return np.asarray(Image.fromarray(image).resize((size, size), Image.LANCZOS))


@dataclass(frozen=True)
class MikasaHistoryConfig:
    history_size: int = 0
    history_stride: int = 1
    obs_history: bool = False
    state_history: bool = False
    history_first_frame: bool = False
    history_obs_views: str = "both"
    history_valid_bits: bool = False
    history_dropout: float = 0.0
    history_time_embedding: bool = False
    history_tokens: bool = False
    history_pool: int = 1

    def __post_init__(self):
        if self.history_size < 0:
            raise ValueError("history_size must be >= 0")
        if self.history_stride < 1:
            raise ValueError("history_stride must be >= 1")
        if self.history_obs_views not in HISTORY_OBS_VIEWS:
            raise ValueError(f"history_obs_views must be one of {HISTORY_OBS_VIEWS}")
        if not 0.0 <= self.history_dropout <= 1.0:
            raise ValueError("history_dropout must be in [0, 1]")
        if (self.obs_history or self.state_history) and self.num_slots == 0:
            raise ValueError("obs/state history needs history_size > 0 or history_first_frame")
        if self.history_time_embedding and not self.obs_history:
            raise ValueError("history_time_embedding needs obs_history")
        if self.history_tokens and not self.obs_history:
            raise ValueError("history_tokens needs obs_history")
        if self.history_tokens and self.history_time_embedding:
            raise ValueError("history_tokens and history_time_embedding are alternative designs; pick one")
        if self.history_pool < 1:
            raise ValueError("history_pool must be >= 1")

    @classmethod
    def from_namespace(cls, namespace) -> "MikasaHistoryConfig":
        return cls(**{name: getattr(namespace, name) for name in cls.__dataclass_fields__})

    @classmethod
    def from_mapping(cls, mapping) -> "MikasaHistoryConfig":
        """Read history fields from e.g. a checkpoint's saved args; missing fields use defaults."""
        return cls(**{name: mapping[name] for name in cls.__dataclass_fields__ if name in mapping})

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def enabled(self) -> bool:
        return self.obs_history or self.state_history

    @property
    def num_slots(self) -> int:
        return int(self.history_first_frame) + self.history_size

    @property
    def views_per_slot(self) -> int:
        return 2 if self.history_obs_views == "both" else 1

    def num_views(self) -> int:
        return 2 + (self.num_slots * self.views_per_slot if self.obs_history else 0)

    def order_views(self, current_views: list, slot_views: list) -> list:
        """Concatenate current and per-slot views in the configured order (see module docstring)."""
        if self.history_time_embedding:
            return [view for views in slot_views for view in views] + list(current_views)
        return list(current_views) + [view for views in slot_views for view in views]

    def view_camera_ids(self) -> list[int]:
        """Camera index per view in model order: 0 = base (third-person), 1 = wrist."""
        slot = [0, 1][: self.views_per_slot]
        slots = slot * (self.num_slots if self.obs_history else 0)
        return self.order_views([0, 1], [slots[i : i + len(slot)] for i in range(0, len(slots), len(slot))])

    def view_time_ids(self) -> list[int]:
        """Time position per view in model order: 0 = oldest slot, num_slots = current step."""
        slots = self.num_slots if self.obs_history else 0
        return self.order_views([slots, slots], [[j] * self.views_per_slot for j in range(slots)])

    @property
    def num_history_views(self) -> int:
        return self.num_slots * self.views_per_slot if self.obs_history else 0

    def state_dim(self, base_state_dim: int) -> int:
        dim = base_state_dim * (1 + (self.num_slots if self.state_history else 0))
        if self.enabled and self.history_valid_bits:
            dim += self.num_slots
        return dim

    def slot_indices(self, t: int, dropout: bool) -> tuple[list[int], list[float]]:
        """Step index and valid bit for each history slot at step t."""
        indices, valid = [], []
        if self.history_first_frame:
            indices.append(t if dropout else 0)
            valid.append(0.0 if dropout else 1.0)
        for k in range(self.history_size, 0, -1):
            index = t - self.history_stride * k
            indices.append(t if dropout else max(index, 0))
            valid.append(0.0 if dropout or index < 0 else 1.0)
        return indices, valid


__all__ = ["HISTORY_OBS_VIEWS", "MikasaHistoryConfig", "resize_camera_image"]
