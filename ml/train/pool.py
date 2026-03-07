import queue
import threading
import torch
import torch.nn.functional as F

from env import MarjapussiEnv
from model import ACTION_FEAT_DIM
from .policy_adjust import apply_bid_consistency_adjustments
from .utils import Log

class BatchInferenceServer:
    """
    Accumulates inference requests from all worker threads and fires one
    GPU forward pass per batch. Expects tensors to be passed in to
    parallelize CPU-heavy conversion.
    """
    def __init__(
        self,
        model,
        device,
        max_batch: int = 128,
        timeout: float = 0.005,
        greedy: bool = False,
        fail_fast: bool = True,
        bid_soft_cap_weight: float = 0.0,
        bid_soft_cap_margin: float = 0.0,
        stop_bid_penalty_weight: float = 0.0,
        stop_bid_margin: float = 0.0,
    ):
        self.model   = model
        self.device  = device
        self.max_batch = max_batch
        self.timeout   = timeout
        self.greedy    = greedy
        self.fail_fast = fail_fast
        self.bid_soft_cap_weight = float(bid_soft_cap_weight)
        self.bid_soft_cap_margin = float(bid_soft_cap_margin)
        self.stop_bid_penalty_weight = float(stop_bid_penalty_weight)
        self.stop_bid_margin = float(stop_bid_margin)
        self._fatal_error: RuntimeError | None = None
        self._req  = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def infer(self, tensor_obs) -> tuple[int, float, float]:
        """Block until inference result is available. Returns (action_idx, value, log_prob). Thread-safe."""
        if self._fatal_error is not None:
            raise RuntimeError("BatchInferenceServer is in fatal state.") from self._fatal_error
        evt  = threading.Event()
        slot = [0, 0.0, 0.0]
        self._req.put((tensor_obs, evt, slot))
        while not evt.wait(timeout=0.5):
            if self._fatal_error is not None:
                raise RuntimeError("BatchInferenceServer failed during inference.") from self._fatal_error
        if self._fatal_error is not None:
            raise RuntimeError("BatchInferenceServer failed during inference.") from self._fatal_error
        return slot[0], slot[1], slot[2]

    def _serve(self):
        while not self._stop.is_set():
            if self._fatal_error is not None:
                while True:
                    try:
                        _, evt, _ = self._req.get_nowait()
                        evt.set()
                    except queue.Empty:
                        break
                return
            items = []
            try:
                items.append(self._req.get(timeout=self.timeout))
                while len(items) < self.max_batch:
                    try: items.append(self._req.get_nowait())
                    except queue.Empty: break
            except queue.Empty:
                continue
            if not items:
                continue
            self._run_batch(items)

    def _run_batch(self, items):
        dev = self.device
        tensors_list = [it[0] for it in items]
        try:
            B = int(len(tensors_list))
            # Quick collate for inference only
            max_s = int(max(int(t["token_ids"].shape[1]) for t in tensors_list))
            max_a = int(max(int(t["action_feats"].shape[1]) for t in tensors_list))
            
            # Batch together and MOVE TO DEVICE
            obs_a = {k: torch.cat([t["obs_a"][k] for t in tensors_list], 0).to(dev)
                     for k in tensors_list[0]["obs_a"]}
            tok = torch.zeros((B, max_s), dtype=torch.long, device=dev)
            tmask = torch.ones((B, max_s), dtype=torch.bool, device=dev)
            for i, t in enumerate(tensors_list):
                L = int(t["token_ids"].shape[1])
                tok[i,:L] = t["token_ids"][0].to(dev, non_blocking=True)
                tmask[i,:L] = t["token_mask"][0].to(dev, non_blocking=True)
            
            af = torch.zeros((B, max_a, ACTION_FEAT_DIM), device=dev)
            am = torch.ones((B, max_a), dtype=torch.bool, device=dev)
            for i, t in enumerate(tensors_list):
                A = t["action_feats"].shape[1]
                af[i,:A] = t["action_feats"][0].to(dev, non_blocking=True)
                am[i,:A] = t["action_mask"][0].to(dev, non_blocking=True)

            with torch.no_grad():
                logits, _, pts_pred, value_pred = self.model({
                    "obs_a":       obs_a,
                    "token_ids":   tok, "token_mask": tmask,
                    "action_feats": af, "action_mask": am,
                })
                masked_logits = logits.masked_fill(am, -1e4)
                masked_logits, _ = apply_bid_consistency_adjustments(
                    masked_logits,
                    af,
                    am,
                    pts_pred,
                    bid_soft_cap_weight=self.bid_soft_cap_weight,
                    bid_soft_cap_margin=self.bid_soft_cap_margin,
                    stop_bid_penalty_weight=self.stop_bid_penalty_weight,
                    stop_bid_margin=self.stop_bid_margin,
                )

            # Keep sampling and log-prob computation on-device; only move final scalars to CPU.
            probs = F.softmax(masked_logits, dim=-1)
            if self.greedy:
                acts = torch.argmax(masked_logits, dim=-1)
            else:
                acts = torch.multinomial(probs, 1).squeeze(1)
            chosen_probs = probs.gather(1, acts.unsqueeze(1)).squeeze(1).clamp_min(1e-8)
            logps = torch.log(chosen_probs)
            values = value_pred.reshape(B)

            acts_cpu = acts.to("cpu").tolist()
            values_cpu = values.to("cpu").tolist()
            logps_cpu = logps.to("cpu").tolist()
            for i, (_, evt, slot) in enumerate(items):
                slot[0] = int(acts_cpu[i])
                slot[1] = float(values_cpu[i])
                slot[2] = float(logps_cpu[i])
                evt.set()
        except Exception as e:
            msg = f"Batch inference failed: {e}"
            Log.error(msg)
            if self.fail_fast:
                self._fatal_error = RuntimeError(msg)
                for _, evt, _slot in items:
                    evt.set()
                return
            for _, evt, slot in items:
                slot[0] = 0
                slot[1] = 0.0
                slot[2] = 0.0
                evt.set()

    def stop(self):
        self._stop.set(); self._thread.join(timeout=1)


class EnvPool:
    """Pool of persistent MarjapussiEnv instances."""
    def __init__(self, size: int, include_labels: bool = False):
        self.envs = queue.Queue()
        for _ in range(size):
            self.envs.put(MarjapussiEnv(include_labels=include_labels))

    def get(self) -> MarjapussiEnv:
        return self.envs.get()

    def put(self, env: MarjapussiEnv):
        self.envs.put(env)

    def close(self):
        while not self.envs.empty():
            self.envs.get().close()
