from __future__ import annotations

import torch


def apply_bid_consistency_adjustments(
    logits: torch.Tensor,
    action_feats: torch.Tensor,
    action_mask: torch.Tensor,
    pts_pred: torch.Tensor,
    *,
    bid_soft_cap_weight: float = 0.0,
    bid_soft_cap_margin: float = 0.0,
    stop_bid_penalty_weight: float = 0.0,
    stop_bid_margin: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Apply action-logit adjustments derived from the model's own team-points estimate.

    This keeps the policy internally consistent without hard-coding a bidding
    heuristic:
    - bids far above predicted achievable value get softened
    - stopping the bidding is discouraged when at least one legal bid looks
      comfortably makeable under the current prediction
    """
    adjusted = logits
    stats: dict[str, torch.Tensor] = {
        "bid_soft_cap_mean": torch.tensor(0.0, device=logits.device),
        "stop_bid_penalty_mean": torch.tensor(0.0, device=logits.device),
        "makeable_bid_rate": torch.tensor(0.0, device=logits.device),
    }

    if action_feats.numel() == 0:
        return adjusted, stats

    is_bid_action = action_feats[:, :, 1] > 0.5
    is_stop_bidding = action_feats[:, :, 2] > 0.5
    legal_mask = ~action_mask
    legal_bid_mask = is_bid_action & legal_mask
    legal_stop_mask = is_stop_bidding & legal_mask

    if not legal_bid_mask.any() and not legal_stop_mask.any():
        return adjusted, stats

    pred_pts_actual = pts_pred[:, 0].unsqueeze(1) * 420.0
    bid_actual = action_feats[:, :, 32] * 300.0 + 120.0

    if bid_soft_cap_weight > 0.0 and legal_bid_mask.any():
        overreach = torch.relu(
            (bid_actual - (pred_pts_actual + float(bid_soft_cap_margin))) / 420.0
        )
        cap_penalty = float(bid_soft_cap_weight) * torch.clamp(overreach * overreach, max=4.0)
        adjusted = adjusted - (cap_penalty * legal_bid_mask.float())
        if legal_bid_mask.any():
            stats["bid_soft_cap_mean"] = cap_penalty[legal_bid_mask].mean()

    if stop_bid_penalty_weight > 0.0 and legal_bid_mask.any() and legal_stop_mask.any():
        makeable_margin = pred_pts_actual - bid_actual + float(stop_bid_margin)
        makeable_score = torch.relu(makeable_margin) / 60.0
        makeable_score = torch.clamp(makeable_score, min=0.0, max=2.0)
        makeable_score = makeable_score * legal_bid_mask.float()
        best_makeable = makeable_score.max(dim=1).values
        stop_penalty = float(stop_bid_penalty_weight) * best_makeable
        adjusted = adjusted - (legal_stop_mask.float() * stop_penalty.unsqueeze(1))
        stats["stop_bid_penalty_mean"] = stop_penalty[legal_stop_mask.any(dim=1)].mean()
        stats["makeable_bid_rate"] = (best_makeable > 0.0).float().mean()

    return adjusted, stats
