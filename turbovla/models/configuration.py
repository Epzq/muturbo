from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass
class TextEncoderConfig:
    model_name_or_path: str = "bert-base-uncased"
    max_length: int = 256
    padding_length: int | None = None
    padding_length_by_instruction: dict[str, int] = field(default_factory=dict)
    sub_sentence_present: bool = True
    frozen: bool = True
    force_eval_when_frozen: bool = True
    zero_padded_tokens: bool = False
    local_files_only: bool = True
    attention_implementation: str | None = None


@dataclass
class VisionEncoderConfig:
    model_name_or_path: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    image_size: int = 256
    num_views: int = 2
    position_embedding: str = "view"
    encode_views_separately: bool = True
    frozen: bool = False
    local_files_only: bool = True
    attention_implementation: str | None = None
    compute_precision: str = "bf16_autocast"
    position_init_std: float = 0.01
    position_scale_init: float = 0.01
    dropout: float = 0.1
    # Optional per-view camera / time indices ("view" position embedding only). When set, each view
    # gets camera_embedding[view_camera_ids[i]] + time_embedding[view_time_ids[i]] instead of a
    # per-view-position embedding. None keeps the original behaviour.
    view_camera_ids: list[int] | None = None
    view_time_ids: list[int] | None = None
    # Optional history tokens: the last ``history_views`` input views are not part of the V-L path;
    # their DINOv3 features (no gradient) are average-pooled to history_pool x history_pool per view,
    # detached, mapped by a separate memory projector and appended to the action decoder memory.
    history_views: int = 0
    history_pool: int = 1


@dataclass
class InteractionConfig:
    hidden_dim: int = 256
    nheads: int = 8
    num_layers: int = 6
    dim_feedforward: int = 2048
    enhancer_inner_dim: int = 1024
    text_dropout: float = 0.0
    fusion_dropout: float = 0.0
    fusion_droppath: float = 0.1
    padding_strategy: str = "key_padding_mask"
    residual_style: str = "normalized"
    attention_backend: str = "manual"
    compute_precision: str = "fp32"


@dataclass
class ActionHeadConfig:
    action_dim: int = 7
    state_dim: int = 8
    horizon: int = 12
    num_state_tokens: int = 2
    num_layers: int = 3
    mlp_hidden_dim: int = 512
    state_hidden_dim: int = 256
    dropout: float = 0.1
    # muTurboVLA recurrent memory (0 = off): number of memory tokens carried across timesteps.
    # They are extra action-decoder queries; their decoder outputs (LayerNorm-ed) become the next
    # step's memory. See docs/MU_TURBOVLA.md.
    memory_tokens: int = 0


@dataclass
class TurboVLAConfig:
    name: str = "TurboVLA"
    compatibility_profile: str = "libero"
    text: TextEncoderConfig = field(default_factory=TextEncoderConfig)
    vision: VisionEncoderConfig = field(default_factory=VisionEncoderConfig)
    interaction: InteractionConfig = field(default_factory=InteractionConfig)
    action: ActionHeadConfig = field(default_factory=ActionHeadConfig)

    def __post_init__(self) -> None:
        if self.name != "TurboVLA":
            raise ValueError(f"model name must be 'TurboVLA', got {self.name!r}")
        if self.compatibility_profile not in {"libero", "robotwin"}:
            raise ValueError(
                "compatibility_profile must be 'libero' or 'robotwin', "
                f"got {self.compatibility_profile!r}"
            )
        if self.vision.num_views < 1:
            raise ValueError("vision.num_views must be positive")
        if self.vision.position_embedding not in {"view", "learned_patch"}:
            raise ValueError("vision.position_embedding must be 'view' or 'learned_patch'")
        camera_ids, time_ids = self.vision.view_camera_ids, self.vision.view_time_ids
        if (camera_ids is None) != (time_ids is None):
            raise ValueError("vision.view_camera_ids and vision.view_time_ids must be set together")
        if camera_ids is not None:
            if len(camera_ids) != self.vision.num_views or len(time_ids) != self.vision.num_views:
                raise ValueError("vision.view_camera_ids / view_time_ids need one entry per view")
            if self.vision.position_embedding != "view":
                raise ValueError("view_camera_ids / view_time_ids require position_embedding='view'")
        if self.vision.history_views:
            if self.vision.history_pool < 1:
                raise ValueError("vision.history_pool must be >= 1")
        if self.vision.compute_precision not in {"fp32", "bf16", "bf16_autocast"}:
            raise ValueError("vision.compute_precision must be fp32, bf16, or bf16_autocast")
        if self.text.padding_length is not None:
            if self.text.padding_length < 1:
                raise ValueError("text.padding_length must be positive")
            if self.text.padding_length > self.text.max_length:
                raise ValueError("text.padding_length cannot exceed text.max_length")
        for instruction, length in self.text.padding_length_by_instruction.items():
            if not instruction:
                raise ValueError("text.padding_length_by_instruction cannot contain an empty instruction")
            if length < 1 or length > self.text.max_length:
                raise ValueError(f"invalid text padding length {length} for instruction {instruction!r}")
        if self.interaction.hidden_dim % self.interaction.nheads != 0:
            raise ValueError("interaction.hidden_dim must be divisible by interaction.nheads")
        if self.interaction.padding_strategy not in {"key_padding_mask", "zero_fill"}:
            raise ValueError("interaction.padding_strategy must be key_padding_mask or zero_fill")
        if self.interaction.residual_style not in {"normalized", "pre_norm"}:
            raise ValueError("interaction.residual_style must be normalized or pre_norm")
        if self.interaction.attention_backend not in {"manual", "sdpa"}:
            raise ValueError("interaction.attention_backend must be manual or sdpa")
        if self.interaction.compute_precision not in {"fp32", "bf16_autocast"}:
            raise ValueError("interaction.compute_precision must be fp32 or bf16_autocast")
        if self.action.action_dim < 1 or self.action.state_dim < 1 or self.action.horizon < 1:
            raise ValueError("action dimensions and horizon must be positive")
        if self.action.memory_tokens < 0:
            raise ValueError("action.memory_tokens must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "TurboVLAConfig":
        data = dict(payload)
        return cls(
            name=str(data.get("name", "TurboVLA")),
            compatibility_profile=str(data.get("compatibility_profile", "libero")),
            text=TextEncoderConfig(**dict(data.get("text", {}))),
            vision=VisionEncoderConfig(**dict(data.get("vision", {}))),
            interaction=InteractionConfig(**dict(data.get("interaction", {}))),
            action=ActionHeadConfig(**dict(data.get("action", {}))),
        )
