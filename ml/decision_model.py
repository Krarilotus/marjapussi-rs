from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

try:
    from ml.model import ACTION_FEAT_DIM
except ModuleNotFoundError:
    from model import ACTION_FEAT_DIM


@dataclass
class DecisionModelConfig:
    card_feature_dim: int = 32
    player_feature_dim: int = 18
    global_feature_dim: int = 15
    model_dim: int = 192
    player_dim: int = 96
    action_dim: int = 128
    dropout: float = 0.05


class DecisionNet(nn.Module):
    def __init__(self, task_name: str, aux_dim: int, cfg: DecisionModelConfig | None = None) -> None:
        super().__init__()
        self.task_name = task_name
        self.cfg = cfg or DecisionModelConfig()
        self.card_tower = nn.Sequential(
            nn.Linear(self.cfg.card_feature_dim, self.cfg.model_dim),
            nn.GELU(),
            nn.Linear(self.cfg.model_dim, self.cfg.model_dim),
            nn.GELU(),
        )
        self.player_tower = nn.Sequential(
            nn.Linear(self.cfg.player_feature_dim, self.cfg.player_dim),
            nn.GELU(),
        )
        self.global_tower = nn.Sequential(
            nn.Linear(self.cfg.global_feature_dim, self.cfg.model_dim),
            nn.GELU(),
        )
        self.action_tower = nn.Sequential(
            nn.Linear(ACTION_FEAT_DIM, self.cfg.action_dim),
            nn.GELU(),
        )
        fused_dim = self.cfg.model_dim + self.cfg.player_dim + self.cfg.action_dim
        self.policy_head = nn.Sequential(
            nn.Linear(fused_dim, self.cfg.model_dim),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.model_dim, 1),
        )
        context_dim = self.cfg.model_dim + self.cfg.player_dim
        self.value_head = nn.Sequential(
            nn.Linear(context_dim, self.cfg.model_dim),
            nn.GELU(),
            nn.Linear(self.cfg.model_dim, 1),
        )
        self.aux_head = nn.Sequential(
            nn.Linear(context_dim, self.cfg.model_dim),
            nn.GELU(),
            nn.Linear(self.cfg.model_dim, aux_dim),
        )

    def forward(
        self,
        card_features: torch.Tensor,
        player_features: torch.Tensor,
        global_features: torch.Tensor,
        action_features: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        card_x = self.card_tower(card_features).mean(dim=1)
        player_x = self.player_tower(player_features).mean(dim=1)
        global_x = self.global_tower(global_features)
        context = torch.cat([card_x + global_x, player_x], dim=-1)

        action_x = self.action_tower(action_features)
        context_rep = context.unsqueeze(1).expand(-1, action_x.shape[1], -1)
        logits = self.policy_head(torch.cat([context_rep, action_x], dim=-1)).squeeze(-1)
        logits = logits.masked_fill(action_mask, -1e4)
        return {
            "policy_logits": logits,
            "value": self.value_head(context).squeeze(-1),
            "aux": self.aux_head(context),
        }

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class BiddingNet(DecisionNet):
    def __init__(self, cfg: DecisionModelConfig | None = None) -> None:
        super().__init__("bidding", aux_dim=3, cfg=cfg)


class PassingNet(DecisionNet):
    def __init__(self, cfg: DecisionModelConfig | None = None) -> None:
        super().__init__("passing", aux_dim=3, cfg=cfg)


class PlayingNet(DecisionNet):
    def __init__(self, cfg: DecisionModelConfig | None = None) -> None:
        super().__init__("playing", aux_dim=3, cfg=cfg)
