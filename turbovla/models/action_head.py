from __future__ import annotations

import torch
from torch import nn

from .components.utils import MLP
from .configuration import ActionHeadConfig


class StateProjection(nn.Module):
    def __init__(self, config: ActionHeadConfig, hidden_dim: int) -> None:
        super().__init__()
        self.num_tokens = int(config.num_state_tokens)
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(config.state_dim),
            nn.Linear(config.state_dim, config.state_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.state_hidden_dim, self.num_tokens * self.hidden_dim),
        )
        self.position = nn.Parameter(torch.randn(1, self.num_tokens, self.hidden_dim) * 0.02)
        self.output_norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim == 3:
            state = state[:, -1]
        if state.ndim != 2:
            raise ValueError(f"state must be [B,D] or [B,T,D], got {tuple(state.shape)}")
        tokens = self.net(state).view(state.shape[0], self.num_tokens, self.hidden_dim)
        return self.output_norm(tokens + self.position.to(device=tokens.device, dtype=tokens.dtype))


class ACTDecoder(nn.Module):
    def __init__(self, config: ActionHeadConfig, hidden_dim: int, nheads: int, dim_feedforward: int) -> None:
        super().__init__()
        self.horizon = int(config.horizon)
        self.num_memory_tokens = int(config.memory_tokens)
        self.action_queries = nn.Embedding(self.horizon, hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.num_layers)
        self.action_projection = MLP(hidden_dim, config.mlp_hidden_dim, config.action_dim, 3)

    def _memory_guard_mask(self, num_memory: int, device: torch.device) -> torch.Tensor:
        """Boolean self-attention mask over [action queries ; memory queries] (True = blocked).

        Memory queries may not attend to action queries (muVLA's guard against copying the predicted
        action chunk into memory); action queries attend to everything.
        """
        size = self.horizon + num_memory
        mask = torch.zeros(size, size, dtype=torch.bool, device=device)
        mask[self.horizon :, : self.horizon] = True
        return mask

    def forward(self, memory: torch.Tensor, memory_queries: torch.Tensor | None = None):
        queries = self.action_queries.weight.unsqueeze(0).expand(memory.shape[0], -1, -1)
        if memory_queries is None:
            hidden = self.decoder(tgt=queries, memory=memory)
            return torch.tanh(self.action_projection(hidden))
        # muTurboVLA: recurrent memory tokens decoded jointly with the action queries.
        queries = torch.cat([queries, memory_queries.to(dtype=queries.dtype)], dim=1)
        mask = self._memory_guard_mask(memory_queries.shape[1], queries.device)
        hidden = self.decoder(tgt=queries, memory=memory, tgt_mask=mask)
        actions = torch.tanh(self.action_projection(hidden[:, : self.horizon]))
        return actions, hidden[:, self.horizon :]


class TurboVLAActionHead(nn.Module):
    def __init__(self, config: ActionHeadConfig, hidden_dim: int, nheads: int, dim_feedforward: int) -> None:
        super().__init__()
        self.state_projection = StateProjection(config, hidden_dim)
        self.decoder = ACTDecoder(config, hidden_dim, nheads, dim_feedforward)
        self.num_memory_tokens = int(config.memory_tokens)
        if self.num_memory_tokens:
            # muVLA-style recurrent memory: learned initial state (M_init), learned memory-position
            # embedding added every step, LayerNorm on the decoder output that becomes M_t.
            self.initial_memory = nn.Parameter(torch.randn(self.num_memory_tokens, hidden_dim) * 0.02)
            self.memory_position = nn.Parameter(torch.randn(1, self.num_memory_tokens, hidden_dim) * 0.02)
            self.memory_norm = nn.LayerNorm(hidden_dim)

    def initial_memory_state(self, batch_size: int) -> torch.Tensor:
        """M_init expanded to [B, m, hidden] (kept in the autograd graph so M_init is learned)."""
        return self.initial_memory.unsqueeze(0).expand(batch_size, -1, -1)

    def forward(
        self,
        vision_language_tokens: torch.Tensor,
        state: torch.Tensor,
        history_tokens: torch.Tensor | None = None,
        recurrent_memory: torch.Tensor | None = None,
        reset_mask: torch.Tensor | None = None,
    ):
        """Returns actions, or (actions, next_memory) when recurrent memory tokens are enabled.

        ``recurrent_memory`` is M_{t-1} [B, m, hidden]; None means episode start (M_init) for the whole
        batch. ``reset_mask`` [B] bool replaces M_{t-1} by M_init where True (episode start per slot).
        """
        state = state.to(device=vision_language_tokens.device, dtype=vision_language_tokens.dtype)
        state_tokens = self.state_projection(state)
        memory = [vision_language_tokens, state_tokens]
        if history_tokens is not None:
            memory.append(history_tokens.to(dtype=vision_language_tokens.dtype))
        context = torch.cat(memory, dim=1)
        if not self.num_memory_tokens:
            return self.decoder(context)
        initial = self.initial_memory_state(context.shape[0]).to(dtype=context.dtype)
        if recurrent_memory is None:
            recurrent_memory = initial
        elif reset_mask is not None:
            recurrent_memory = torch.where(reset_mask[:, None, None], initial, recurrent_memory.to(dtype=context.dtype))
        queries = recurrent_memory.to(dtype=context.dtype) + self.memory_position.to(dtype=context.dtype)
        actions, memory_hidden = self.decoder(context, memory_queries=queries)
        return actions, self.memory_norm(memory_hidden)
