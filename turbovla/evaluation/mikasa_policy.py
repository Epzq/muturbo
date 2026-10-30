"""TurboVLA policy adapter for the MIKASA-Robo-VLA benchmark runner.

The MIKASA runner (``mikasa_robo_suite.vla.benchmarking``) calls ``policy.forward(obs)``
only when its action queue is empty. History slots need every environment step, so this
adapter reports ``chunk_size = 1`` to the runner and keeps the model's action chunk in its
own queue: ``forward`` runs once per step, records the observation, and re-plans when the
internal queue is empty.

Inputs are rebuilt exactly as in training (turbovla/data/mikasa_rlds.py):
  * DINOv3 preprocessing without resize/crop at the checkpoint's image size,
  * views = [base_t, wrist_t, (base_s, wrist_s) for s in history slots],
  * state = [s_t, (s_s for s in slots), (valid_s for s in slots)], z-scored with the
    training stats; slots come from the same ``MikasaHistoryConfig.slot_indices``.
Actions undo the training min/max normalization for dims 0-5; the continuous gripper
command (dim 6) is passed through, not binarized.

The runner never calls ``reset``; call ``start_task``/``reset`` from its task/episode
callbacks (experiments/mikasa/evaluate.py does this).
"""

from __future__ import annotations

import json
import os
from collections import deque
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor

from ..data.mikasa_history import MikasaHistoryConfig, resize_camera_image

HISTORY_EVAL_MODES = ("normal", "current", "shuffled")
REPO_ROOT = Path(__file__).resolve().parents[2]


def _select_stats(payload: Mapping[str, Any], stats_key: str | None) -> Mapping[str, Any]:
    if stats_key:
        if stats_key not in payload:
            raise KeyError(f"stats_key={stats_key!r} not found in stats payload")
        return payload[stats_key]
    keys = [key for key in payload if key != "metadata"]
    if len(keys) != 1:
        raise KeyError(f"stats_key is required; available keys: {keys}")
    return payload[keys[0]]


def _resolve_path(path: str) -> str:
    """Training stores paths relative to the TurboVLA repo root; resolve them from anywhere."""
    if os.path.isabs(path) or os.path.exists(path):
        return path
    candidate = REPO_ROOT / path
    return str(candidate) if candidate.exists() else path


def _to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class TurboVLAMikasaPolicy:
    def __init__(
        self,
        ckpt_path: str,
        dinov3_path: str | None = None,
        bert_path: str | None = None,
        stats_path: str | None = None,
        stats_key: str | None = None,
        device: str | torch.device | None = None,
        precision: str = "bf16",
        num_open_loop_steps: int | None = None,
        history_eval_mode: str = "normal",
        seed: int = 0,
        allow_hf_download: bool = False,
        verbose: bool = True,
    ) -> None:
        if precision not in {"bf16", "fp32"}:
            raise ValueError(f"precision must be 'bf16' or 'fp32', got {precision!r}")
        if history_eval_mode not in HISTORY_EVAL_MODES:
            raise ValueError(f"history_eval_mode must be one of {HISTORY_EVAL_MODES}")

        self.ckpt_path = str(ckpt_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.precision = precision
        self.model_dtype = torch.bfloat16 if precision == "bf16" else torch.float32
        self.history_eval_mode = history_eval_mode
        self.verbose = verbose
        self._rng = np.random.default_rng(seed)

        checkpoint = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        args = dict(checkpoint.get("args") or {})
        if "model_config" not in checkpoint or "ema_model_state_dict" not in checkpoint:
            raise KeyError("checkpoint must contain `model_config` and `ema_model_state_dict`")
        self.train_args = args
        self.global_step = checkpoint.get("global_step")
        self.history = MikasaHistoryConfig.from_mapping(args)
        self.resize_images_to = int(args.get("resize_images_to") or 0)

        self.dinov3_path = _resolve_path(dinov3_path or args["dinov3_path"])
        self.bert_path = bert_path or args["bert_path"]
        local_files_only = not allow_hf_download and not args.get("allow_hf_download", False)
        self.model = self._build_model(checkpoint, local_files_only, precision)
        self.model.load_state_dict(
            {key.removeprefix("module."): value for key, value in checkpoint["ema_model_state_dict"].items()},
            strict=True,
        )
        del checkpoint
        if precision == "bf16":
            self.model.to(dtype=torch.bfloat16)
        self.model.to(self.device).eval().requires_grad_(False)

        config = self.model.config
        self.model_chunk_size = int(config.action.horizon)
        # muTurboVLA recurrent memory: the model runs at every env step to update M_t.
        self.memory_tokens = int(getattr(config.action, "memory_tokens", 0))
        self.num_views = int(config.vision.num_views)
        self.image_size = int(config.vision.image_size)
        self.state_dim = int(config.action.state_dim)
        if self.resize_images_to and self.resize_images_to != self.image_size:
            raise ValueError(f"checkpoint resize_images_to={self.resize_images_to} != model image_size={self.image_size}")
        model_views = self.num_views + int(config.vision.history_views)  # history_tokens: current + history
        if model_views != self.history.num_views():
            raise ValueError(f"model takes {model_views} views but history config implies {self.history.num_views()}")
        self.num_open_loop_steps = int(num_open_loop_steps or self.model_chunk_size)
        if not 1 <= self.num_open_loop_steps <= self.model_chunk_size:
            raise ValueError(f"num_open_loop_steps must be in [1, {self.model_chunk_size}]")

        self._load_stats(_resolve_path(stats_path or args["stats_path"]), stats_key or args.get("stats_key"), args)
        expected_state_dim = self.history.state_dim(self.proprio_mean.size)
        if expected_state_dim != self.state_dim:
            raise ValueError(f"model state_dim={self.state_dim} but stats/history imply {expected_state_dim}")

        self.processor = AutoImageProcessor.from_pretrained(self.dinov3_path, local_files_only=local_files_only)
        for name in ("do_resize", "do_center_crop"):
            if hasattr(self.processor, name):
                setattr(self.processor, name, False)

        # The benchmark runner pops one action per forward call; the real chunk lives in self._queue.
        self.chunk_size = 1
        self.instruction = ""
        self.reset()
        if verbose:
            print(
                f"[TurboVLAMikasaPolicy] ckpt={self.ckpt_path} step={self.global_step} "
                f"views={self.num_views} state_dim={self.state_dim} image_size={self.image_size} "
                f"resize_images_to={self.resize_images_to or 'off'} "
                f"chunk={self.model_chunk_size} open_loop={self.num_open_loop_steps} "
                f"history={self.history.to_dict()} history_eval_mode={history_eval_mode}",
                flush=True,
            )

    # ------------------------------------------------------------------ setup

    def _build_model(self, checkpoint: Mapping[str, Any], local_files_only: bool, precision: str):
        from ..models.configuration import TurboVLAConfig
        from ..models.turbovla import build_turbovla

        config = TurboVLAConfig.from_mapping(checkpoint["model_config"])
        config.text.model_name_or_path = self.bert_path
        config.text.local_files_only = local_files_only
        config.vision.model_name_or_path = self.dinov3_path
        config.vision.local_files_only = local_files_only
        # Same as the LIBERO adapter: pure bf16 DINOv3 for bf16 eval, bf16 autocast for fp32 eval.
        config.vision.compute_precision = "bf16" if precision == "bf16" else "bf16_autocast"
        return build_turbovla(config)

    def _load_stats(self, stats_path: str, stats_key: str | None, args: Mapping[str, Any]) -> None:
        with open(stats_path, encoding="utf-8") as f:
            stats = _select_stats(json.load(f), stats_key)
        state_section = "proprio" if "proprio" in stats else "state"
        self.proprio_mean = np.asarray(stats[state_section]["mean"], dtype=np.float32)
        self.proprio_std = np.asarray(stats[state_section]["std"], dtype=np.float32)
        self.action_min = np.asarray(stats["action"]["min"], dtype=np.float32)
        self.action_max = np.asarray(stats["action"]["max"], dtype=np.float32)
        # Same rule as LiberoMixedRLDSDataset.
        mode = str(args.get("normalize_binary_gripper", "auto"))
        if mode == "auto":
            self.normalize_binary_gripper = bool(self.action_min[6] >= -1e-6 and self.action_max[6] <= 1.0 + 1e-6)
        else:
            self.normalize_binary_gripper = mode in {"1", "true", "yes"}

    # --------------------------------------------------------------- episodes

    def start_task(self, instruction: str) -> None:
        self.instruction = str(instruction)
        self.reset()

    def reset(self) -> None:
        self._frames: list[tuple[np.ndarray, np.ndarray]] = []
        self._states: list[np.ndarray] = []
        self._pixels: dict[tuple[int, int], torch.Tensor] = {}
        self._queue: deque[np.ndarray] = deque()
        self._memory: torch.Tensor | None = None  # recurrent memory M_{t-1}; None = episode start (M_init)
        # Per-episode prediction log (see episode_record).
        self._pred_steps: list[int] = []
        self._pred_norm: list[np.ndarray] = []
        self._pred_env: list[np.ndarray] = []
        self._executed: list[np.ndarray] = []

    # ------------------------------------------------------------------ inputs

    def _record(self, obs: Mapping[str, Any]) -> None:
        rgb = _to_numpy(obs["rgb"])
        rgb = rgb.reshape(-1, *rgb.shape[-3:])[0]  # (H, W, 6): base camera, then wrist camera
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        base, wrist = np.ascontiguousarray(rgb[..., :3]), np.ascontiguousarray(rgb[..., 3:6])
        if self.resize_images_to:
            # Same resize as training (MikasaProprioMixin._process_image_pair).
            base = resize_camera_image(base, self.resize_images_to)
            wrist = resize_camera_image(wrist, self.resize_images_to)
        if base.shape[:2] != (self.image_size, self.image_size):
            raise ValueError(f"expected {self.image_size}x{self.image_size} camera images, got {base.shape[:2]}")
        self._frames.append((base, wrist))
        proprio = _to_numpy(obs["proprio"]).reshape(-1)[: self.proprio_mean.size].astype(np.float32)
        self._states.append(proprio)

    def _pixel_values(self, step: int, camera: int) -> torch.Tensor:
        key = (step, camera)
        if key not in self._pixels:
            image = Image.fromarray(self._frames[step][camera])
            self._pixels[key] = self.processor(images=image, return_tensors="pt")["pixel_values"][0]
        return self._pixels[key]

    def _slots(self, t: int) -> tuple[list[int], list[float]]:
        if self.history_eval_mode == "current":
            return self.history.slot_indices(t, dropout=True)
        indices, valid = self.history.slot_indices(t, dropout=False)
        if self.history_eval_mode == "shuffled" and len(indices) > 1:
            order = self._rng.permutation(len(indices))
            indices = [indices[i] for i in order]
            valid = [valid[i] for i in order]
        return indices, valid

    def _model_inputs(self) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        t = len(self._frames) - 1
        current_views = [self._pixel_values(t, 0), self._pixel_values(t, 1)]
        slot_views = []
        state_parts = [self._states[t]]
        if self.history.enabled:
            indices, valid = self._slots(t)
            if self.history.obs_history:
                for index in indices:
                    slot_views.append([self._pixel_values(index, camera) for camera in range(self.history.views_per_slot)])
            if self.history.state_history:
                state_parts.extend(self._states[index] for index in indices)
        normalized = [(state - self.proprio_mean) / (self.proprio_std + 1e-6) for state in state_parts]
        if self.history.enabled and self.history.history_valid_bits:
            normalized.append(np.asarray(valid, dtype=np.float32))
        state = torch.from_numpy(np.concatenate(normalized)).unsqueeze(0)
        views = self.history.order_views(current_views, slot_views)
        samples = {"dinov3": torch.stack(views, dim=0).unsqueeze(0)}
        return (
            {key: value.to(self.device, dtype=self.model_dtype) for key, value in samples.items()},
            state.to(self.device, dtype=self.model_dtype),
        )

    # ----------------------------------------------------------------- actions

    def _denormalize(self, chunk: np.ndarray) -> np.ndarray:
        actions = chunk.astype(np.float32).copy()
        span = self.action_max[:6] - self.action_min[:6] + 1e-6
        actions[:, :6] = 0.5 * (actions[:, :6] + 1.0) * span + self.action_min[:6]
        if self.normalize_binary_gripper:
            actions[:, 6] = 0.5 * (actions[:, 6] + 1.0)
        return np.clip(actions, -1.0, 1.0)

    @torch.inference_mode()
    def _run_model(self) -> np.ndarray:
        """One model call on the current step; updates the recurrent memory if enabled.

        Returns the normalized action chunk. history_eval_mode="current" disables memory for recurrent
        models (M_init at every step), the memory analogue of replacing history by the current step.
        """
        samples, state = self._model_inputs()
        if self.memory_tokens:
            memory_in = None if self.history_eval_mode == "current" else self._memory
            pred, self._memory = self.model([self.instruction], samples, state, recurrent_memory=memory_in)
        else:
            pred = self.model([self.instruction], samples, state)
        return np.nan_to_num(pred.float().cpu().numpy()[0], nan=0.0, posinf=1.0, neginf=-1.0)

    def _log_prediction(self, chunk: np.ndarray) -> np.ndarray:
        env_chunk = self._denormalize(chunk)
        self._pred_steps.append(len(self._frames) - 1)
        self._pred_norm.append(chunk)
        self._pred_env.append(env_chunk)
        return env_chunk

    def predict_chunk(self) -> np.ndarray:
        return self._log_prediction(self._run_model())

    def forward(self, obs: Mapping[str, Any]) -> torch.Tensor:
        self._record(obs)
        if self.memory_tokens:
            # Memory must be updated every step (as in training), even while executing a queued chunk.
            chunk = self._run_model()
            if not self._queue:
                self._queue.extend(self._log_prediction(chunk)[: self.num_open_loop_steps])
        elif not self._queue:
            self._queue.extend(self.predict_chunk()[: self.num_open_loop_steps])
        action = torch.from_numpy(self._queue.popleft())
        self._executed.append(action.numpy().copy())
        proprio = obs.get("proprio")
        return action.to(proprio.device) if torch.is_tensor(proprio) else action

    def episode_record(self) -> dict[str, np.ndarray]:
        """Arrays for the episode so far; call before reset().

        proprio (T, 7): raw observed proprio per env step; executed_actions (T, 7): action sent each
        step (MIKASA's no-op wrapper still zeroes it during cue/wait phases); pred_steps (N,): env
        step of each model call; pred_chunks_norm / pred_chunks_env (N, H, 7): model output in
        training normalization and as env actions.
        """
        horizon, dim = self.model_chunk_size, self.action_min.size
        return {
            "proprio": np.asarray(self._states, dtype=np.float32).reshape(-1, self.proprio_mean.size),
            "executed_actions": np.asarray(self._executed, dtype=np.float32).reshape(-1, dim),
            "pred_steps": np.asarray(self._pred_steps, dtype=np.int64),
            "pred_chunks_norm": np.asarray(self._pred_norm, dtype=np.float32).reshape(-1, horizon, dim),
            "pred_chunks_env": np.asarray(self._pred_env, dtype=np.float32).reshape(-1, horizon, dim),
        }

    def describe(self) -> dict[str, Any]:
        return {
            "name": "turbovla",
            "config": {
                "ckpt_path": self.ckpt_path,
                "global_step": self.global_step,
                "model_chunk_size": self.model_chunk_size,
                "num_open_loop_steps": self.num_open_loop_steps,
                "precision": self.precision,
                "history_eval_mode": self.history_eval_mode,
                "history": self.history.to_dict(),
                "num_views": self.num_views,
                "memory_tokens": self.memory_tokens,
                "state_dim": self.state_dim,
                "stats_key": self.train_args.get("stats_key"),
                "resize_images_to": self.resize_images_to,
            },
        }


__all__ = ["HISTORY_EVAL_MODES", "TurboVLAMikasaPolicy"]
