"""MIKASA-Robo-VLA RLDS dataset mixed across multiple TFDS tasks.

The released MIKASA-Robo-VLA RLDS matches the LIBERO layout (``observation/image``,
``observation/wrist_image``, 7-D ``action``, per-step ``language_instruction``) except
that proprioception is stored as ``observation/proprio`` instead of ``observation/state``.

Optional observation/state history ("history slots"):

    slots = [first frame (optional)] + [t - stride*K, ..., t - stride]   (oldest -> newest)

* obs history appends each slot's camera views after the current views, so the model
  sees ``num_views = 2 + num_slots * views_per_slot`` views; slot identity comes from
  the per-view learned embedding.
* state history appends each slot's normalized state after the current state.
* Window slots before the episode start repeat frame/state 0.
* history_valid_bits (ablation, off by default) appends one bit per slot: 1 for a real
  past step, 0 for padding or dropout.
* history_dropout replaces all slots with the current step for a random fraction of
  samples, i.e. "act as if this were the episode start".

Sample layout:
    views = [base_t, wrist_t, (base_s, wrist_s) for s in slots]   # wrist omitted for "base"
    state = [s_t, (s_s for s in slots), (valid_s for s in slots)]
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import get_worker_info

from .libero_rlds import vla_collate_fn
from .mikasa_history import HISTORY_OBS_VIEWS, MikasaHistoryConfig, resize_camera_image
from .mixed_suite import LiberoMixedRLDSDataset


def raw_state(observation):
    return observation["state"] if "state" in observation else observation["proprio"]


class MikasaProprioMixin:
    """Expose MIKASA ``observation/proprio`` under the ``state`` key the LIBERO loader reads.

    With ``resize_images_to`` > 0, every camera image (current and history) is resized to that
    size before DINOv3 preprocessing; ``expected_image_size`` must then equal it.
    """

    def __init__(self, *args, resize_images_to: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.resize_images_to = int(resize_images_to or 0)
        if self.resize_images_to and self.resize_images_to != self.expected_image_size:
            raise ValueError(
                f"resize_images_to={self.resize_images_to} must equal expected_image_size={self.expected_image_size}"
            )

    def _process_image_pair(self, img_np):
        if self.resize_images_to:
            img_np = resize_camera_image(img_np, self.resize_images_to)
        return super()._process_image_pair(img_np)

    def _build_step_sample(self, steps, t, episode_len):
        observation = steps[t]["observation"]
        if "state" not in observation:
            observation["state"] = observation["proprio"]
        return super()._build_step_sample(steps, t, episode_len)


class MikasaHistoryMixin(MikasaProprioMixin):
    """Append observation and/or state history slots to each sample (see module docstring)."""

    def __init__(self, *args, history: MikasaHistoryConfig | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.history = history or MikasaHistoryConfig()
        self._history_rng = None

    def _rng(self):
        if self._history_rng is None:
            worker_info = get_worker_info()
            worker_id = 0 if worker_info is None else worker_info.id
            self._history_rng = np.random.default_rng(self.seed + 7919 * self.rank + 104729 * worker_id + 1)
        return self._history_rng

    def _build_step_sample(self, steps, t, episode_len):
        images, instruction, state, action_chunk, action_chunk_mask = super()._build_step_sample(steps, t, episode_len)
        history = self.history
        if not history.enabled:
            return images, instruction, state, action_chunk, action_chunk_mask

        dropout = history.history_dropout > 0.0 and self._rng().random() < history.history_dropout
        indices, valid = history.slot_indices(t, dropout)

        slot_views = []
        if history.obs_history:
            camera_keys = ("image", "wrist_image")[: history.views_per_slot]
            processed = {t: images}
            for index in indices:
                if index not in processed:
                    observation = steps[index]["observation"]
                    processed[index] = tuple(self._process_image_pair(observation[key]) for key in camera_keys)
                slot_views.append(processed[index][: history.views_per_slot])
        views = history.order_views(images, slot_views)

        state_parts = [state]
        if history.state_history:
            for index in indices:
                past = torch.tensor(raw_state(steps[index]["observation"]), dtype=torch.float32)
                state_parts.append(self._normalize_state(past))
        if history.history_valid_bits:
            state_parts.append(torch.tensor(valid, dtype=torch.float32))

        return tuple(views), instruction, torch.cat(state_parts), action_chunk, action_chunk_mask


class MikasaRLDSDataset(MikasaHistoryMixin, LiberoMixedRLDSDataset):
    pass


def mikasa_collate_fn(batch):
    """Like vla_collate_fn, but for any fixed number of views per sample."""
    if len(batch) == 0:
        raise ValueError("empty batch cannot be collated")
    num_views = len(batch[0][0])
    views = [[] for _ in range(num_views)]
    instructions, states, action_chunks, action_chunk_masks = [], [], [], []
    for images, instruction, state, action_chunk, action_chunk_mask in batch:
        if len(images) != num_views:
            raise ValueError(f"expected {num_views} views per sample, got {len(images)}")
        for view_idx, image in enumerate(images):
            views[view_idx].append(image["dinov3"])
        instructions.append(instruction)
        states.append(state)
        action_chunks.append(action_chunk)
        action_chunk_masks.append(action_chunk_mask)

    samples = {"dinov3": torch.stack([torch.stack(view, dim=0) for view in views], dim=1)}
    return (
        samples,
        instructions,
        torch.stack(states, dim=0),
        torch.stack(action_chunks, dim=0),
        torch.stack(action_chunk_masks, dim=0),
    )


__all__ = [
    "HISTORY_OBS_VIEWS",
    "MikasaHistoryConfig",
    "MikasaHistoryMixin",
    "MikasaProprioMixin",
    "MikasaRLDSDataset",
    "mikasa_collate_fn",
    "raw_state",
    "vla_collate_fn",
]
