from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from action_slots import MAX_ACTIONS

INPUT_DIM = 510


@dataclass(frozen=True)
class ModelConfig:
    input_dim: int = INPUT_DIM
    hidden_dim: int = 512
    bottleneck_dim: int = 256
    max_actions: int = MAX_ACTIONS
    dropout_p: float = 0.10


class DeepCfrNet(nn.Module):
    """5x512 + LayerNorm network used for both advantage and strategy heads."""

    def __init__(self, cfg: ModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or ModelConfig()

        self.fc1 = nn.Linear(self.cfg.input_dim, self.cfg.hidden_dim)
        self.ln1 = nn.LayerNorm(self.cfg.hidden_dim)
        self.fc2 = nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim)
        self.ln2 = nn.LayerNorm(self.cfg.hidden_dim)
        self.fc3 = nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim)
        self.ln3 = nn.LayerNorm(self.cfg.hidden_dim)
        self.fc4 = nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim)
        self.ln4 = nn.LayerNorm(self.cfg.hidden_dim)
        self.fc5 = nn.Linear(self.cfg.hidden_dim, self.cfg.bottleneck_dim)
        self.ln5 = nn.LayerNorm(self.cfg.bottleneck_dim)
        self.fc_out = nn.Linear(self.cfg.bottleneck_dim, self.cfg.max_actions)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=self.cfg.dropout_p)

    def _mask_logits(
        self,
        logits: torch.Tensor,
        action_mask: torch.Tensor | None,
        fill_value: float,
    ) -> torch.Tensor:
        if action_mask is None:
            return logits

        if action_mask.ndim == 1:
            # Backward compatibility for count-based callers.
            valid_actions = action_mask.to(dtype=torch.long, device=logits.device)
            indices = torch.arange(self.cfg.max_actions, device=logits.device).unsqueeze(0)
            mask = indices < valid_actions.unsqueeze(1)
        elif action_mask.ndim == 2 and action_mask.shape[1] == self.cfg.max_actions:
            if action_mask.dtype == torch.bool:
                mask = action_mask.to(device=logits.device)
            else:
                mask = action_mask.to(device=logits.device) > 0.5
        else:
            raise ValueError(
                f"action_mask must be shape [B] or [B,{self.cfg.max_actions}], got {tuple(action_mask.shape)}"
            )
        safe_fill_value = fill_value
        if torch.is_floating_point(logits):
            safe_fill_value = max(fill_value, float(torch.finfo(logits.dtype).min))
        fill = torch.full_like(logits, safe_fill_value)
        return torch.where(mask, logits, fill)

    def forward(
        self,
        x: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        strategy_mode: bool = False,
    ) -> torch.Tensor:
        x = self.dropout(self.relu(self.ln1(self.fc1(x))))
        x = self.dropout(self.relu(self.ln2(self.fc2(x))))
        x = self.dropout(self.relu(self.ln3(self.fc3(x))))
        x = self.dropout(self.relu(self.ln4(self.fc4(x))))
        x = self.dropout(self.relu(self.ln5(self.fc5(x))))
        logits = self.fc_out(x)

        if strategy_mode:
            logits = self._mask_logits(logits, action_mask, fill_value=-1e9)
            return torch.softmax(logits, dim=-1)

        logits = self._mask_logits(logits, action_mask, fill_value=0.0)
        return logits
