import os
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L


class VQTrainer(L.LightningModule):
    """
    Train a VectorQuantize codebook on mel representations.

    Loss: uses VectorQuantize's returned loss (commitment + codebook EMA updates handled internally).

    Notes:
    - If your mels dim != codeword_dims, we learn a linear projection into VQ space.
    - We ignore padded frames when computing loss (masked mean).
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

        self.mel_dim = int(config["mel_dim"])              # e.g. n_mels
        self.vq_dim = int(config["codeword_dims"])         # codeword embedding dim
        self.num_codewords = int(config["num_codewords"])

        # optional: keep input magnitude stable; cosine sim is used in your earlier setup
        self.use_cosine_sim = bool(config.get("use_cosine_sim", True))

        # projection (minimal) if mel_dim != codeword_dims
        self.in_proj = nn.Identity() if self.mel_dim == self.vq_dim else nn.Linear(self.mel_dim, self.vq_dim, bias=False)

        from vector_quantize_pytorch import VectorQuantize
        self.vq = VectorQuantize(
            dim=self.vq_dim,
            codebook_size=self.num_codewords,
            decay=float(config.get("ema_decay", 0.7)),
            commitment_weight=float(config.get("beta", 0.1)),
            kmeans_init=bool(config.get("kmeans_init", True)),
            kmeans_iters=int(config.get("kmeans_iters", 10)),
            use_cosine_sim=self.use_cosine_sim,
            threshold_ema_dead_code=int(config.get("threshold_ema_dead_code", 2)),
            orthogonal_reg_weight=float(config.get("ortho_weight", 0.0)),
            orthogonal_reg_active_codes_only=bool(config.get("orthogonal_reg_active_codes_only", False)),
        )

        # training hyperparams
        self.optim = config.get("optimizer", "adam")
        self.lr = float(config.get("lr", 5e-4))
        self.wt_decay = float(config.get("wt_decay", 0.0))
        self.lr_scheduler = bool(config.get("lr_scheduler", False))

        self.save_hyperparameters(config)

    @staticmethod
    def _pick_first_view(x):
        # supports batch["utterances"] being (anc,pos) or single tensor
        if isinstance(x, (list, tuple)):
            return x[0]
        return x

    @staticmethod
    def _pick_padmask(batch: Dict[str, Any]) -> Optional[torch.Tensor]:
        """
        Returns padmask with True=pad shape (B,T), if present.
        Supports batch["utterance_idx"] being list/tuple of 2 or single.
        """
        if "utterance_idx" not in batch:
            return None
        uidx = batch["utterance_idx"]
        if isinstance(uidx, (list, tuple)):
            return uidx[0].bool()
        return uidx.bool()

    def _masked_mean(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        # x: (B,T) or (B,T,*) reduced already; valid: (B,T) bool
        x = x[valid]
        return x.mean() if x.numel() > 0 else x.sum() * 0.0

    def _shared_step(self, batch: Dict[str, Any], mode: str) -> torch.Tensor:
        mels = self._pick_first_view(batch["utterances"])  # (B,T,F)
        padmask = self._pick_padmask(batch)                # (B,T) True=pad or None

        z = self.in_proj(mels)                             # (B,T,vq_dim)

        # VectorQuantize expects (B, T, D) by default in vector-quantize-pytorch
        quantized, indices, vq_loss = self.vq(z)

        # Mask padding: vq_loss is typically scalar already (averaged).
        # But if your VectorQuantize returns per-token loss in your version,
        # handle it here robustly.
        if padmask is not None:
            valid = ~padmask
            # if vq_loss is scalar -> keep it
            if vq_loss.dim() > 0:
                # try to reduce any per-token loss shapes
                # common: (B,T) or (B,T,1)
                while vq_loss.dim() > 2:
                    vq_loss = vq_loss.squeeze(-1)
                if vq_loss.dim() == 2:
                    vq_loss = self._masked_mean(vq_loss, valid)
                else:
                    vq_loss = vq_loss.mean()
        else:
            vq_loss = vq_loss.mean() if vq_loss.dim() > 0 else vq_loss

        # Track code usage / perplexity-ish stats
        with torch.no_grad():
            # indices: (B,T)
            if padmask is not None:
                idx = indices[~padmask]
            else:
                idx = indices.reshape(-1)
            if idx.numel() > 0:
                used = torch.unique(idx).numel()
                usage = used / float(self.num_codewords)
            else:
                usage = torch.tensor(0.0, device=self.device)

        self.log_dict(
            {
                f"{mode}/loss": vq_loss.detach(),
                f"{mode}/code_usage_frac": usage.detach(),
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return vq_loss

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        loss = self._shared_step(batch, mode="train")
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        _ = self._shared_step(batch, mode="valid")

    def configure_optimizers(self):
        if self.optim.lower() == "adam":
            opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optim.lower() == "adamw":
            opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wt_decay)
        else:
            raise NotImplementedError(f"Unknown optimizer: {self.optim}")

        if not self.lr_scheduler:
            return {"optimizer": opt}

        # minimal cosine warmup scheduler (same style you used earlier)
        sched = transformers.get_cosine_schedule_with_warmup(
            opt,
            num_warmup_steps=int(self.config.get("warmup_steps", 1000)),
            num_training_steps=int(self.config.get("max_steps", 100000)),
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step"},
        }


