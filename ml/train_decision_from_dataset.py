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

from decision_model import BiddingNet, DecisionModelConfig, PassingNet, PlayingNet
from checkpoint_utils import load_model_checkpoint
from pretrain_cache import load_or_build_decision_cache
from decision_state import (
    TASK_AUX_TARGET_NAMES,
)
from train.utils import Log


DEFAULT_CKPT_DIR = Path(__file__).parent / "checkpoints"
TASK_TO_MODEL = {
    "bidding": BiddingNet,
    "passing": PassingNet,
    "playing": PlayingNet,
}


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


def collate_decision(records: list[dict]) -> dict[str, torch.Tensor] | None:
    card_feats = []
    player_feats = []
    global_feats = []
    action_feats = []
    action_masks = []
    policy_targets = []
    value_targets = []
    aux_targets = []
    sample_weights = []
    teacher_policies = []

    for sample in records:
        card_feats.append(sample["card_features"])
        player_feats.append(sample["player_features"])
        global_feats.append(sample["global_features"])
        action_feats.append(sample["action_features"])
        action_masks.append(sample["action_mask"])
        policy_targets.append(int(sample["policy_target"]))
        value_targets.append(float(sample["value_target"]))
        aux_targets.append(sample["aux_targets"])
        sample_weights.append(float(sample["sample_weight"]))
        teacher_policies.append(sample["teacher_policy"])

    if not card_feats:
        return None

    max_actions = max(t.shape[0] for t in action_feats)
    action_feat_dim = action_feats[0].shape[1]
    padded_action_feats = []
    padded_action_masks = []
    padded_teacher_policies = []
    for feats, mask, teacher_policy in zip(action_feats, action_masks, teacher_policies):
        pad_actions = max_actions - feats.shape[0]
        if pad_actions > 0:
            feats = F.pad(feats, (0, 0, 0, pad_actions))
            mask = F.pad(mask, (0, pad_actions), value=True)
        if teacher_policy is None:
            teacher_policy = torch.zeros(feats.shape[0], dtype=torch.float32)
        else:
            teacher_policy = teacher_policy.to(dtype=torch.float32)
            if teacher_policy.shape[0] > feats.shape[0]:
                teacher_policy = teacher_policy[: feats.shape[0]]
            elif teacher_policy.shape[0] < feats.shape[0]:
                teacher_policy = F.pad(teacher_policy, (0, feats.shape[0] - teacher_policy.shape[0]))
        padded_action_feats.append(feats)
        padded_action_masks.append(mask)
        padded_teacher_policies.append(teacher_policy)

    return {
        "card_features": torch.stack(card_feats),
        "player_features": torch.stack(player_feats),
        "global_features": torch.stack(global_feats),
        "action_features": torch.stack(padded_action_feats).reshape(len(padded_action_feats), max_actions, action_feat_dim),
        "action_mask": torch.stack(padded_action_masks),
        "policy_targets": torch.tensor(policy_targets, dtype=torch.long),
        "value_targets": torch.tensor(value_targets, dtype=torch.float32),
        "aux_targets": torch.stack(aux_targets),
        "sample_weights": torch.tensor(sample_weights, dtype=torch.float32),
        "teacher_policy": torch.stack(padded_teacher_policies),
    }


def _slice_batch(payload: dict, indices: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "card_features": payload["card_features"].index_select(0, indices),
        "player_features": payload["player_features"].index_select(0, indices),
        "global_features": payload["global_features"].index_select(0, indices),
        "action_features": payload["action_features"].index_select(0, indices),
        "action_mask": payload["action_mask"].index_select(0, indices),
        "policy_targets": payload["policy_targets"].index_select(0, indices),
        "value_targets": payload["value_targets"].index_select(0, indices),
        "aux_targets": payload["aux_targets"].index_select(0, indices),
        "sample_weights": payload["sample_weights"].index_select(0, indices),
        "teacher_policy": payload["teacher_policy"].index_select(0, indices),
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


def build_model(task: str) -> torch.nn.Module:
    return TASK_TO_MODEL[task](DecisionModelConfig())


def train(
    data_path: str,
    task: str,
    epochs: int = 3,
    batch: int = 256,
    lr: float = 3e-4,
    device: str = "cpu",
    workers: int = 4,
    checkpoints_dir: str | Path = DEFAULT_CKPT_DIR,
    log_every: int = 200,
    max_steps: int = 0,
    min_epochs: int = 1,
    target_acc: float = 0.0,
    target_acc_streak: int = 2,
    no_amp: bool = False,
    checkpoint: str | Path | None = None,
    ignore_checkpoint_epoch_budget: bool = False,
) -> dict[str, float]:
    if task not in TASK_TO_MODEL:
        raise ValueError(f"unsupported task '{task}'")
    configure_torch_runtime(device, workers)
    ckpt_dir = Path(checkpoints_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(task).to(device)
    start_epoch = 0
    global_step = 0
    if checkpoint is not None:
        payload = load_model_checkpoint(model, checkpoint, device=device, expected_task=task)
        metadata = dict(payload.get("metadata", {}))
        start_epoch = int(metadata.get("epochs_seen", metadata.get("epoch", 0)))
        global_step = int(metadata.get("global_step", 0))
        Log.info(f"Resumed {task} checkpoint: {checkpoint}")
        if ignore_checkpoint_epoch_budget:
            start_epoch = 0
    Log.success(
        f"Decision pretraining | task={task} | epochs={epochs} | batch={batch} | workers={workers} | device={device}"
    )
    Log.info(f"Dataset path: {data_path}")
    Log.info(f"Model params: {model.param_count():,}")
    Log.info(f"Aux targets: {', '.join(TASK_AUX_TARGET_NAMES[task])}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = GradScaler("cuda", enabled=(device.startswith("cuda") and not no_amp))

    try:
        cache_path, cached_payload = load_or_build_decision_cache(data_path, task)
        Log.info(f"Cached dataset: {cache_path}")
        Log.info(f"Samples: {int(cached_payload['sample_count']):,}")
        Log.info("Packed tensor cache: enabled (workers ignored for human pretraining)")
    except Exception as exc:
        raise RuntimeError(f"failed to build decision cache for {task}: {exc}") from exc

    steps_per_epoch = max(1, ceil(int(cached_payload["sample_count"]) / max(1, batch)))

    remaining_epochs = max(0, int(epochs) - start_epoch)
    if remaining_epochs <= 0:
        Log.info(
            f"{task.capitalize()} checkpoint already reached requested epoch budget "
            f"({start_epoch}/{epochs}); skipping further training."
        )
        return {
            "accuracy": float(metadata.get("accuracy", 0.0)) if checkpoint is not None else 0.0,
            "epochs_seen": float(start_epoch),
            "global_step": float(global_step),
        }

    best_acc = -1.0
    streak = 0
    last_summary: dict[str, float] = {}
    overall_t0 = time.time()

    for epoch_idx in range(remaining_epochs):
        epochs_seen = start_epoch + epoch_idx + 1
        Log.phase(
            f"{task.capitalize()} Epoch {epochs_seen} (+{epoch_idx + 1}/{remaining_epochs})"
        )
        model.train()
        step = 0
        sum_loss = 0.0
        sum_policy = 0.0
        sum_value = 0.0
        sum_aux = 0.0
        sum_teacher = 0.0
        correct = 0
        total = 0
        t0 = time.time()
        stop_early = False

        for step, batch_data in _iter_packed_batches(cached_payload, batch, shuffle=True):
            batch_data = _move_batch_to_device(batch_data, device)
            weights = batch_data["sample_weights"]

            with autocast("cuda", enabled=(device.startswith("cuda") and not no_amp)):
                outputs = model(
                    card_features=batch_data["card_features"],
                    player_features=batch_data["player_features"],
                    global_features=batch_data["global_features"],
                    action_features=batch_data["action_features"],
                    action_mask=batch_data["action_mask"],
                )
                per_policy_loss = F.cross_entropy(
                    outputs["policy_logits"],
                    batch_data["policy_targets"],
                    reduction="none",
                )
                policy_loss = (per_policy_loss * weights).sum() / weights.sum().clamp(min=1.0)
                teacher_policy = batch_data["teacher_policy"]
                if teacher_policy.sum().item() > 0.0:
                    teacher_log_probs = F.log_softmax(outputs["policy_logits"], dim=-1)
                    teacher_loss = -(
                        teacher_policy * teacher_log_probs
                    ).sum(dim=-1).mean()
                else:
                    teacher_loss = outputs["policy_logits"].new_tensor(0.0)
                value_loss = F.mse_loss(outputs["value"], batch_data["value_targets"])
                aux_loss = F.mse_loss(outputs["aux"], batch_data["aux_targets"])
                loss = policy_loss + 0.25 * value_loss + 0.20 * aux_loss + 0.35 * teacher_loss

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            with torch.no_grad():
                preds = outputs["policy_logits"].argmax(dim=-1)
                correct += int((preds == batch_data["policy_targets"]).sum().item())
                total += int(batch_data["policy_targets"].numel())

            global_step += 1
            sum_loss += float(loss.item())
            sum_policy += float(policy_loss.item())
            sum_value += float(value_loss.item())
            sum_aux += float(aux_loss.item())
            sum_teacher += float(teacher_loss.item())

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
                    f"Loss: {sum_loss / step:.4f} | Pol: {sum_policy / step:.4f} | "
                    f"Val: {sum_value / step:.4f} | Aux: {sum_aux / step:.4f} | "
                    f"Teach: {sum_teacher / step:.4f} | "
                    f"Acc: {(correct / max(1, total)):.3f} | {samples_per_sec:,.0f} samples/s | "
                    f"ETA: {_format_duration(eta_seconds)}",
                    end="",
                )

            if max_steps > 0 and global_step >= max_steps:
                stop_early = True
                break

        if step > 0:
            print()
        accuracy = correct / max(1, total)
        payload = {
            "state_dict": model.state_dict(),
            "metadata": {
                "task": task,
                "model_family": getattr(model, "task_name", task),
                "epoch": epochs_seen,
                "epochs_seen": epochs_seen,
                "global_step": global_step,
                "accuracy": accuracy,
                "policy_loss": sum_policy / max(1, step),
                "value_loss": sum_value / max(1, step),
                "aux_loss": sum_aux / max(1, step),
                "teacher_loss": sum_teacher / max(1, step),
                "aux_targets": TASK_AUX_TARGET_NAMES[task],
            },
        }
        latest_path = ckpt_dir / f"{task}_latest.pt"
        best_path = ckpt_dir / f"{task}_best.pt"
        torch.save(payload, latest_path)
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(payload, best_path)

        Log.success(f"{task.capitalize()} Epoch {epochs_seen} Summary:")
        print(f"  - Steps:     {step}")
        print(f"  - Accuracy:  {accuracy:.4f}")
        epoch_elapsed = time.time() - t0
        total_elapsed = time.time() - overall_t0
        remaining_epoch_count = max(0, remaining_epochs - epoch_idx - 1)
        print(f"  - Time:      Epoch: {_format_duration(epoch_elapsed)} | RunETA: {_format_duration(remaining_epoch_count * epoch_elapsed)} | Elapsed: {_format_duration(total_elapsed)}")
        print(
            f"  - Losses:    Total: {sum_loss / max(1, step):.4f} | "
            f"Pol: {sum_policy / max(1, step):.4f} | "
            f"Val: {sum_value / max(1, step):.4f} | Aux: {sum_aux / max(1, step):.4f} | "
            f"Teach: {sum_teacher / max(1, step):.4f}"
        )
        print(f"  - Saved:     {latest_path}")
        last_summary = {
            "accuracy": accuracy,
            "epochs_seen": float(epochs_seen),
            "global_step": float(global_step),
            "policy_loss": sum_policy / max(1, step),
            "value_loss": sum_value / max(1, step),
            "aux_loss": sum_aux / max(1, step),
            "teacher_loss": sum_teacher / max(1, step),
        }

        if target_acc > 0.0 and epochs_seen >= max(1, int(min_epochs)):
            if accuracy >= target_acc:
                streak += 1
                if streak >= max(1, int(target_acc_streak)):
                    Log.success(
                        f"Stopping on accuracy target after epoch {epochs_seen}: "
                        f"{accuracy:.4f} >= {target_acc:.4f}"
                    )
                    break
            else:
                streak = 0
        if stop_early:
            break

    Log.success(f"{task.capitalize()} pretraining complete.")
    Log.info(f"Checkpoint: {ckpt_dir / f'{task}_latest.pt'}")
    return last_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--task", choices=sorted(TASK_TO_MODEL.keys()), required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--checkpoints-dir", default=str(DEFAULT_CKPT_DIR))
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--min-epochs", type=int, default=1)
    parser.add_argument("--target-acc", type=float, default=0.0)
    parser.add_argument("--target-acc-streak", type=int, default=2)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    train(
        data_path=args.data,
        task=args.task,
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        device=args.device,
        workers=args.workers,
        checkpoints_dir=args.checkpoints_dir,
        log_every=args.log_every,
        max_steps=args.max_steps,
        min_epochs=args.min_epochs,
        target_acc=args.target_acc,
        target_acc_streak=args.target_acc_streak,
        no_amp=args.no_amp,
        checkpoint=args.checkpoint,
    )
