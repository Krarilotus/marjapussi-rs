from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

try:
    from ml.neurosymbolic_dataset import NUM_OWNER_CLASSES
except ModuleNotFoundError:
    from neurosymbolic_dataset import NUM_OWNER_CLASSES


@dataclass
class BeliefModelConfig:
    card_feature_dim: int = 23
    player_feature_dim: int = 18
    global_feature_dim: int = 12
    model_dim: int = 192
    player_dim: int = 96
    depth: int = 3
    dropout: float = 0.05


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class BeliefNet(nn.Module):
    model_family = "belief_v1"

    def __init__(self, cfg: BeliefModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or BeliefModelConfig()

        self.card_in = nn.Sequential(
            nn.Linear(self.cfg.card_feature_dim + self.cfg.global_feature_dim, self.cfg.model_dim),
            nn.GELU(),
        )
        self.player_in = nn.Sequential(
            nn.Linear(self.cfg.player_feature_dim + self.cfg.global_feature_dim, self.cfg.player_dim),
            nn.GELU(),
        )
        self.card_blocks = nn.ModuleList(
            [ResidualBlock(self.cfg.model_dim, self.cfg.dropout) for _ in range(self.cfg.depth)]
        )
        self.player_blocks = nn.ModuleList(
            [ResidualBlock(self.cfg.player_dim, self.cfg.dropout) for _ in range(max(1, self.cfg.depth - 1))]
        )

        fused_dim = self.cfg.model_dim + self.cfg.player_dim
        self.card_fuse = nn.Sequential(
            nn.Linear(fused_dim, self.cfg.model_dim),
            nn.GELU(),
            nn.Linear(self.cfg.model_dim, NUM_OWNER_CLASSES),
        )
        self.player_void_head = nn.Linear(self.cfg.player_dim, 4)
        self.player_half_head = nn.Linear(self.cfg.player_dim, 4)
        self.player_pair_head = nn.Linear(self.cfg.player_dim, 4)

    def forward(
        self,
        card_features: torch.Tensor,
        player_features: torch.Tensor,
        global_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size, card_count, _ = card_features.shape
        _, player_count, _ = player_features.shape

        global_card = global_features.unsqueeze(1).expand(batch_size, card_count, -1)
        global_player = global_features.unsqueeze(1).expand(batch_size, player_count, -1)

        card_x = self.card_in(torch.cat([card_features, global_card], dim=-1))
        player_x = self.player_in(torch.cat([player_features, global_player], dim=-1))

        for block in self.card_blocks:
            card_x = block(card_x)
        for block in self.player_blocks:
            player_x = block(player_x)

        pooled_players = player_x.mean(dim=1, keepdim=True).expand(-1, card_count, -1)
        card_logits = self.card_fuse(torch.cat([card_x, pooled_players], dim=-1))

        return {
            "card_logits": card_logits,
            "player_void_logits": self.player_void_head(player_x),
            "player_half_logits": self.player_half_head(player_x),
            "player_pair_logits": self.player_pair_head(player_x),
        }

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
