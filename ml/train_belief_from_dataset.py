from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from math import ceil

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

sys.path.insert(0, str(Path(__file__).parent))

from belief_model import BeliefModelConfig, BeliefNet
from checkpoint_utils import load_model_checkpoint
from pretrain_cache import load_or_build_belief_cache
from train.utils import Log


DEFAULT_CKPT_DIR = Path(__file__).parent / "checkpoints"


def _mean_abs_error(logits: torch.Tensor, targets: torch.Tensor) -> float:
    probs = torch.sigmoid(logits)
    return float((probs - targets).abs().mean().item())


def configure_torch_runtime(device: str, workers: int) -> None:
    torch.set_float32_matmul_precision("high")
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    cpu_count = os.cpu_count() or 4
    if workers > 0:
        torch.set_num_threads(max(1, min(8, cpu_count // max(1, workers))))


def _format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def collate_belief(records: list[dict]) -> dict[str, torch.Tensor] | None:
    card_feats = []
    player_feats = []
    global_feats = []
    card_targets = []
    hidden_masks = []
    candidate_masks = []
    void_targets = []
    half_targets = []
    pair_targets = []

    for sample in records:
        card_feats.append(sample["card_features"])
        player_feats.append(sample["player_features"])
        global_feats.append(sample["global_features"])
        card_targets.append(sample["card_targets"])
        hidden_masks.append(sample["hidden_mask"])
        candidate_masks.append(sample["candidate_mask"])
        void_targets.append(sample["void_targets"])
        half_targets.append(sample["half_targets"])
        pair_targets.append(sample["pair_targets"])

    if not card_feats:
        return None

    return {
        "card_features": torch.stack(card_feats),
        "player_features": torch.stack(player_feats),
        "global_features": torch.stack(global_feats),
        "card_targets": torch.stack(card_targets),
        "hidden_mask": torch.stack(hidden_masks),
        "candidate_mask": torch.stack(candidate_masks),
        "void_targets": torch.stack(void_targets),
        "half_targets": torch.stack(half_targets),
        "pair_targets": torch.stack(pair_targets),
    }


def _slice_batch(payload: dict, indices: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "card_features": payload["card_features"].index_select(0, indices),
        "player_features": payload["player_features"].index_select(0, indices),
        "global_features": payload["global_features"].index_select(0, indices),
        "card_targets": payload["card_targets"].index_select(0, indices),
        "hidden_mask": payload["hidden_mask"].index_select(0, indices),
        "candidate_mask": payload["candidate_mask"].index_select(0, indices),
        "void_targets": payload["void_targets"].index_select(0, indices),
        "half_targets": payload["half_targets"].index_select(0, indices),
        "pair_targets": payload["pair_targets"].index_select(0, indices),
    }


def _iter_packed_batches(payload: dict, batch_size: int, *, shuffle: bool) -> tuple[int, dict[str, torch.Tensor]]:
    sample_count = int(payload["sample_count"])
    order = torch.randperm(sample_count) if shuffle else torch.arange(sample_count)
    step = 0
    for start in range(0, sample_count, batch_size):
        step += 1
        indices = order[start : start + batch_size]
        yield step, _slice_batch(payload, indices)


def _move_batch_to_device(batch_data: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch_data.items()
    }


def train(
    data_path: str,
    epochs: int = 3,
    batch: int = 256,
    lr: float = 3e-4,
    device: str = "cpu",
    workers: int = 4,
    checkpoints_dir: str | Path = DEFAULT_CKPT_DIR,
    log_every: int = 200,
    max_steps: int = 0,
    min_epochs: int = 1,
    target_hidden_acc: float = 0.0,
    target_hidden_streak: int = 2,
    no_amp: bool = False,
    checkpoint: str | Path | None = None,
    ignore_checkpoint_epoch_budget: bool = False,
) -> dict[str, float]:
    configure_torch_runtime(device, workers)
    ckpt_dir = Path(checkpoints_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = BeliefNet(BeliefModelConfig()).to(device)
    start_epoch = 0
    if checkpoint is not None:
        payload = load_model_checkpoint(model, checkpoint, device=device)
        metadata = dict(payload.get("metadata", {}))
        start_epoch = int(metadata.get("epochs_seen", metadata.get("epoch", 0)))
        Log.info(f"Resumed belief checkpoint: {checkpoint}")
        if ignore_checkpoint_epoch_budget:
            start_epoch = 0
    Log.success(
        f"Belief pretraining | epochs={epochs} | batch={batch} | workers={workers} | device={device}"
    )
    Log.info(f"Dataset path: {data_path}")
    Log.info(f"Model params: {model.param_count():,}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = GradScaler("cuda", enabled=(device.startswith("cuda") and not no_amp))

    try:
        cache_path, cached_payload = load_or_build_belief_cache(data_path)
        Log.info(f"Cached dataset: {cache_path}")
        Log.info(f"Samples: {int(cached_payload['sample_count']):,}")
        Log.info("Packed tensor cache: enabled (workers ignored for human pretraining)")
    except Exception as exc:
        raise RuntimeError(f"failed to build belief cache: {exc}") from exc

    steps_per_epoch = max(1, ceil(int(cached_payload["sample_count"]) / max(1, batch)))

    global_step = 0
    if checkpoint is not None:
        global_step = int(metadata.get("global_step", 0))
    remaining_epochs = max(0, int(epochs) - start_epoch)
    if remaining_epochs <= 0:
        Log.info(
            "Belief checkpoint already reached requested epoch budget "
            f"({start_epoch}/{epochs}); skipping further training."
        )
        return {
            "card_owner_acc": float(metadata.get("hidden_accuracy", 0.0)) if checkpoint is not None else 0.0,
            "constraint_consistency": float(
                metadata.get("belief_metrics", {}).get("constraint_consistency", 1.0)
            )
            if checkpoint is not None
            else 1.0,
            "epochs_seen": float(start_epoch),
            "global_step": float(global_step),
        }

    best_hidden_acc = -1.0
    streak = 0
    min_epochs = max(1, int(min_epochs))
    target_hidden_streak = max(1, int(target_hidden_streak))
    overall_t0 = time.time()

    last_metrics: dict[str, float] = {}
    for epoch_idx in range(remaining_epochs):
        epochs_seen = start_epoch + epoch_idx + 1
        Log.phase(f"Belief Epoch {epochs_seen} (+{epoch_idx + 1}/{remaining_epochs})")
        model.train()
        step = 0
        sum_loss = 0.0
        sum_card = 0.0
        sum_aux = 0.0
        correct_hidden = 0
        total_hidden = 0
        correct_void = 0
        total_void = 0
        correct_half = 0
        total_half = 0
        correct_pair = 0
        total_pair = 0
        calibration_sum = 0.0
        calibration_count = 0
        t0 = time.time()
        stop_early = False

        for step, batch_data in _iter_packed_batches(cached_payload, batch, shuffle=True):
            batch_data = _move_batch_to_device(batch_data, device)
            with autocast("cuda", enabled=(device.startswith("cuda") and not no_amp)):
                outputs = model(
                    card_features=batch_data["card_features"],
                    player_features=batch_data["player_features"],
                    global_features=batch_data["global_features"],
                )

                masked_logits = outputs["card_logits"].masked_fill(~batch_data["candidate_mask"], -1e4)
                per_card_loss = F.cross_entropy(
                    masked_logits.reshape(-1, masked_logits.shape[-1]),
                    batch_data["card_targets"].reshape(-1),
                    reduction="none",
                ).reshape_as(batch_data["card_targets"])

                hidden_weights = batch_data["hidden_mask"].float()
                exact_weights = (~batch_data["hidden_mask"]).float() * 0.25
                weights = hidden_weights + exact_weights
                card_loss = (per_card_loss * weights).sum() / weights.sum().clamp(min=1.0)

                void_loss = F.binary_cross_entropy_with_logits(
                    outputs["player_void_logits"], batch_data["void_targets"]
                )
                half_loss = F.binary_cross_entropy_with_logits(
                    outputs["player_half_logits"], batch_data["half_targets"]
                )
                pair_loss = F.binary_cross_entropy_with_logits(
                    outputs["player_pair_logits"], batch_data["pair_targets"]
                )
                aux_loss = void_loss + half_loss + pair_loss
                loss = card_loss + 0.25 * aux_loss

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            with torch.no_grad():
                preds = masked_logits.argmax(dim=-1)
                hidden_mask = batch_data["hidden_mask"]
                correct_hidden += int(((preds == batch_data["card_targets"]) & hidden_mask).sum().item())
                total_hidden += int(hidden_mask.sum().item())
                void_preds = (torch.sigmoid(outputs["player_void_logits"]) >= 0.5)
                half_preds = (torch.sigmoid(outputs["player_half_logits"]) >= 0.5)
                pair_preds = (torch.sigmoid(outputs["player_pair_logits"]) >= 0.5)
                correct_void += int((void_preds == (batch_data["void_targets"] >= 0.5)).sum().item())
                total_void += int(batch_data["void_targets"].numel())
                correct_half += int((half_preds == (batch_data["half_targets"] >= 0.5)).sum().item())
                total_half += int(batch_data["half_targets"].numel())
                correct_pair += int((pair_preds == (batch_data["pair_targets"] >= 0.5)).sum().item())
                total_pair += int(batch_data["pair_targets"].numel())
                hidden_probs = masked_logits.softmax(dim=-1)
                hidden_truth = hidden_probs.gather(
                    -1, batch_data["card_targets"].unsqueeze(-1)
                ).squeeze(-1)
                if hidden_mask.any():
                    hidden_calibration = float(hidden_truth[hidden_mask].mean().item())
                    calibration_sum += hidden_calibration
                    calibration_count += 1
                calibration_sum += 1.0 - _mean_abs_error(outputs["player_void_logits"], batch_data["void_targets"])
                calibration_sum += 1.0 - _mean_abs_error(outputs["player_half_logits"], batch_data["half_targets"])
                calibration_sum += 1.0 - _mean_abs_error(outputs["player_pair_logits"], batch_data["pair_targets"])
                calibration_count += 3

            global_step += 1
            sum_loss += float(loss.item())
            sum_card += float(card_loss.item())
            sum_aux += float(aux_loss.item())

            if step % max(1, min(log_every, max(1, steps_per_epoch // 8))) == 0:
                elapsed = time.time() - t0
                samples_per_sec = (step * batch) / max(elapsed, 1e-6)
                remaining_steps = max(0, steps_per_epoch - step)
                remaining_epoch_steps = max(0, remaining_epochs - epoch_idx - 1)
                eta_seconds = (remaining_steps + remaining_epoch_steps * steps_per_epoch) * (
                    elapsed / max(1, step)
                )
                Log.opt(
                    f"Epoch {epochs_seen} (+{epoch_idx + 1}/{remaining_epochs}) | Step {step}/{steps_per_epoch} | "
                    f"Loss: {sum_loss / step:.4f} | Card: {sum_card / step:.4f} | "
                    f"Aux: {sum_aux / step:.4f} | HiddenAcc: "
                    f"{(correct_hidden / max(1, total_hidden)):.3f} | {samples_per_sec:,.0f} samples/s | "
                    f"ETA: {_format_duration(eta_seconds)}",
                    end="",
                )

            if max_steps > 0 and global_step >= max_steps:
                stop_early = True
                break

        if step > 0:
            print()
        hidden_acc = correct_hidden / max(1, total_hidden)
        void_acc = correct_void / max(1, total_void)
        half_acc = correct_half / max(1, total_half)
        pair_acc = correct_pair / max(1, total_pair)
        half_pair_acc = 0.5 * (half_acc + pair_acc)
        calibration_score = calibration_sum / max(1, calibration_count)
        constraint_consistency = 1.0
        payload = {
            "state_dict": model.state_dict(),
            "metadata": {
                "model_family": model.model_family,
                "epoch": epochs_seen,
                "epochs_seen": epochs_seen,
                "global_step": global_step,
                "hidden_accuracy": hidden_acc,
                "belief_metrics": {
                    "card_owner_acc": hidden_acc,
                    "constraint_consistency": constraint_consistency,
                    "void_suit_acc": void_acc,
                    "half_pair_acc": half_pair_acc,
                    "pair_acc": pair_acc,
                    "calibration_score": calibration_score,
                },
            },
        }
        torch.save(payload, ckpt_dir / "belief_latest.pt")
        if hidden_acc > best_hidden_acc:
            best_hidden_acc = hidden_acc
            torch.save(payload, ckpt_dir / "belief_best.pt")

        Log.success(f"Belief Epoch {epochs_seen} Summary:")
        print(f"  - Steps:     {step}")
        print(f"  - HiddenAcc: {hidden_acc:.4f}")
        epoch_elapsed = time.time() - t0
        total_elapsed = time.time() - overall_t0
        remaining_epoch_count = max(0, remaining_epochs - epoch_idx - 1)
        print(f"  - Time:      Epoch: {_format_duration(epoch_elapsed)} | RunETA: {_format_duration(remaining_epoch_count * epoch_elapsed)} | Elapsed: {_format_duration(total_elapsed)}")
        print(
            f"  - BeliefQ:   VoidAcc: {void_acc:.4f} | HalfAcc: {half_acc:.4f} | "
            f"PairAcc: {pair_acc:.4f} | Calib: {calibration_score:.4f}"
        )
        print(f"  - Losses:    Total: {sum_loss / max(1, step):.4f} | Card: {sum_card / max(1, step):.4f} | Aux: {sum_aux / max(1, step):.4f}")
        print(f"  - Saved:     {ckpt_dir / 'belief_latest.pt'}")
        last_metrics = dict(payload["metadata"]["belief_metrics"])
        last_metrics["epochs_seen"] = float(epochs_seen)
        last_metrics["global_step"] = float(global_step)

        if target_hidden_acc > 0.0 and epochs_seen >= min_epochs:
            if hidden_acc >= target_hidden_acc:
                streak += 1
                if streak >= target_hidden_streak:
                    Log.success(
                        f"Stopping on hidden-accuracy target after epoch {epochs_seen}: "
                        f"{hidden_acc:.4f} >= {target_hidden_acc:.4f}"
                    )
                    break
            else:
                streak = 0
        if stop_early:
            break

    Log.success("Belief pretraining complete.")
    Log.info(f"Checkpoint: {ckpt_dir / 'belief_latest.pt'}")
    return last_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--checkpoints-dir", default=str(DEFAULT_CKPT_DIR))
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--min-epochs", type=int, default=1)
    parser.add_argument("--target-hidden-acc", type=float, default=0.0)
    parser.add_argument("--target-hidden-streak", type=int, default=2)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    train(
        data_path=args.data,
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        device=args.device,
        workers=args.workers,
        checkpoints_dir=args.checkpoints_dir,
        log_every=args.log_every,
        max_steps=args.max_steps,
        min_epochs=args.min_epochs,
        target_hidden_acc=args.target_hidden_acc,
        target_hidden_streak=args.target_hidden_streak,
        no_amp=args.no_amp,
        checkpoint=args.checkpoint,
    )
