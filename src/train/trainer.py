import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torchaudio.transforms as T

import lightning as L
import transformers

# -------------------------
# NEW: minimal helper modules
# -------------------------

class WordBiLSTMEncoder(nn.Module):
    """
    Input:  (B, T, F)  where F = mel feature dim (already extracted)
    Output: word embedding (B, D) from last valid timestep of BiLSTM outputs
    """
    def __init__(self, in_dim: int, hidden: int, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out_dim = 2 * hidden

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, F)
        lengths: (B,) number of valid frames per sample
        """
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)  # (B, T, 2H)

        # last valid timestep: out[b, lengths[b]-1]
        idx = (lengths - 1).clamp(min=0).view(-1, 1, 1).to(out.device)  # (B,1,1)
        idx = idx.expand(-1, 1, out.size(-1))                           # (B,1,2H)
        last = out.gather(dim=1, index=idx).squeeze(1)                  # (B,2H)
        return last


class TokenSeqDecoder(nn.Module):
    """
    Decode a source word embedding into token logits aligned to a target time axis.

    Given src_emb (B, D), produce logits (B, T_tgt, V).
    Minimal: repeat src_emb across time and run a small BiLSTM + linear head.
    """
    def __init__(self, emb_dim: int, hidden: int, vocab: int, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.proj = nn.Linear(2 * hidden, vocab)

    def forward(self, src_emb: torch.Tensor, tgt_lengths: torch.Tensor, max_t: int) -> torch.Tensor:
        """
        src_emb: (B, D)
        tgt_lengths: (B,)
        max_t: int (max target length in batch)
        """
        B, D = src_emb.shape
        x = src_emb.unsqueeze(1).expand(B, max_t, D)  # (B, T, D)

        packed = nn.utils.rnn.pack_padded_sequence(
            x, tgt_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=max_t)  # (B,T,2H)

        logits = self.proj(out)  # (B,T,V)
        return logits




def load_pretrained_vq_into_model(
    model,                      # your AudioWordEmbedding
    vq_ckpt_path: str,
    device: str = "cuda",
):
    # 1) load checkpoint dict
    ckpt = torch.load(vq_ckpt_path, map_location=device)
    sd = ckpt["state_dict"]  # Lightning state_dict

    # 2) copy VQ weights

    vq_src = {k.replace("vq.", "vq_layer.", 1): v for k, v in sd.items() if k.startswith("vq.")}
    missing, unexpected = model.load_state_dict(vq_src, strict=False)

    # 3) (IMPORTANT) copy projection if you used it (mel_dim != codeword_dims)
    proj_src = {k.replace("in_proj.", "mel_to_vq.", 1): v for k, v in sd.items() if k.startswith("in_proj.")}
    m2, u2 = model.load_state_dict(proj_src, strict=False)

    # 4) freeze + eval
    model.vq_layer.eval()
    for p in model.vq_layer.parameters():
        p.requires_grad = False

    # freeze projection too (optional but usually desired)
    model.mel_to_vq.eval()
    for p in model.mel_to_vq.parameters():
        p.requires_grad = False

    print("Loaded VQ. Missing:", missing, "Unexpected:", unexpected)
    print("Loaded proj. Missing:", m2, "Unexpected:", u2)
    return model


class AudioWordEmbedding(L.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config

        # -------------------------
        # IMPORTANT CHANGE:
        # we now use a BiLSTM to generate word embeddings from mel features
        # -------------------------
        mel_dim = self.config["mel_dim"]              # <-- MUST match your dataloader feature dim
        emb_hidden = self.config["bilstm_hidden"]     # e.g., 256
        emb_layers = self.config.get("bilstm_layers", 1)
        emb_dropout = self.config.get("bilstm_dropout", 0.0)

        self.word_encoder = WordBiLSTMEncoder(
            in_dim=mel_dim,
            hidden=emb_hidden,
            num_layers=emb_layers,
            dropout=emb_dropout,
        )
        word_emb_dim = self.word_encoder.out_dim

        # -------------------------
        # VQ: pretrained, kept in eval + frozen
        # Assumption: VQ consumes mel features directly (B,T,mel_dim) or (B,T,codeword_dims).
        # If your VQ expects codeword_dims != mel_dim, we add a minimal linear projection.
        # -------------------------
        self.vq_in_dim = self.config["codeword_dims"]
        if self.vq_in_dim != mel_dim:
            self.mel_to_vq = nn.Linear(mel_dim, self.vq_in_dim, bias=False)
        else:
            self.mel_to_vq = nn.Identity()

        self.vq_layer = VectorQuantize(
            dim=self.config["codeword_dims"],
            codebook_size=self.config["num_codewords"],
            decay=self.config["ema_decay"],
            commitment_weight=self.config["beta"],
            kmeans_init=True,
            kmeans_iters=10,
            use_cosine_sim=True,
            threshold_ema_dead_code=2,
            orthogonal_reg_weight=self.config["ortho_weight"],
            orthogonal_reg_active_codes_only=False,
        )

        # Freeze VQ parameters (pretrained / tokenization only)
        self.vq_layer.eval()
        for p in self.vq_layer.parameters():
            p.requires_grad = False

        # -------------------------
        # Decoder: src word embedding -> token logits aligned with target token sequence
        # One shared decoder used both directions (anc->pos and pos->anc)
        # -------------------------
        dec_hidden = self.config.get("dec_hidden", 256)
        dec_layers = self.config.get("dec_layers", 1)
        dec_dropout = self.config.get("dec_dropout", 0.0)

        self.decoder = TokenSeqDecoder(
            emb_dim=word_emb_dim,
            hidden=dec_hidden,
            vocab=self.config["num_codewords"],
            num_layers=dec_layers,
            dropout=dec_dropout,
        )

        self.temperature = self.config["temperature"]
        self.cost_factors = self.config["cost_factors"]
        self.optim = self.config["optimizer"]
        self.lr = self.config["lr"]
        self.wt_decay = self.config["wt_decay"]
        self.lr_scheduler = self.config["lr_scheduler"]
        self.add_augment = self.config["add_augment"]

        self.save_hyperparameters()

    def _lengths_from_padmask(self, padmask: torch.Tensor) -> torch.Tensor:
        """
        padmask: (B,T) boolean where True means "is padding" (based on your earlier usage)
        returns lengths: (B,) number of valid frames
        """
        # valid = ~padmask
        lengths = (~padmask.bool()).sum(dim=1)
        return lengths.clamp(min=1)

    def _ce_loss_time_masked(self, logits: torch.Tensor, targets: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        logits:  (B, T, V)
        targets: (B, T) int64
        lengths: (B,)
        """
        B, T, V = logits.shape
        device = logits.device

        # mask valid timesteps
        t = torch.arange(T, device=device).unsqueeze(0).expand(B, T)  # (B,T)
        valid = t < lengths.unsqueeze(1)                              # (B,T)

        # flatten
        logits_f = logits.reshape(B * T, V)
        targets_f = targets.reshape(B * T)
        valid_f = valid.reshape(B * T)

        # compute CE only on valid positions
        loss = F.cross_entropy(logits_f[valid_f], targets_f[valid_f])
        return loss

    def _shared_step(self, batch, mode):
        log_values = {}

        # Dataloader already provides mel features
        anc = batch["utterances"][0]  # (B,T,F)
        pos = batch["utterances"][1]  # (B,T,F)

        # pad masks (based on your earlier code)
        anc_pad = batch["utterance_idx"][0].bool()  # (B,T) True=pad
        pos_pad = batch["utterance_idx"][1].bool()  # (B,T)

        anc_len = self._lengths_from_padmask(anc_pad)
        pos_len = self._lengths_from_padmask(pos_pad)

        # 1) Word embeddings from BiLSTM (last valid timestep)
        anc_word = self.word_encoder(anc, anc_len)  # (B,D)
        pos_word = self.word_encoder(pos, pos_len)  # (B,D)

        # 2) VQ tokens from mel features (pretrained VQ in eval, no grad)
        with torch.no_grad():
            anc_for_vq = self.mel_to_vq(anc)  # (B,T,codeword_dims)
            pos_for_vq = self.mel_to_vq(pos)

            # vector-quantize-pytorch returns (quantized, indices, loss) typically
            _, indices_anc, _ = self.vq_layer(anc_for_vq)  # indices: (B,T)
            _, indices_pos, _ = self.vq_layer(pos_for_vq)

        # 3) Decode anc_word -> pos token logits aligned to pos timeline
        T_pos = pos.size(1)
        logits_anc2pos = self.decoder(src_emb=anc_word, tgt_lengths=pos_len, max_t=T_pos)  # (B,T_pos,V)

        # 4) Decode pos_word -> anc token logits aligned to anc timeline
        T_anc = anc.size(1)
        logits_pos2anc = self.decoder(src_emb=pos_word, tgt_lengths=anc_len, max_t=T_anc)  # (B,T_anc,V)

        # 5) Cross-entropy losses (time-masked)
        loss_a2p = self._ce_loss_time_masked(logits_anc2pos, indices_pos, pos_len)
        loss_p2a = self._ce_loss_time_masked(logits_pos2anc, indices_anc, anc_len)
        loss = 0.5 * (loss_a2p + loss_p2a)

        log_values[f"{mode}/loss"] = loss.detach()
        log_values[f"{mode}/loss_a2p"] = loss_a2p.detach()
        log_values[f"{mode}/loss_p2a"] = loss_p2a.detach()

        self.log_dict(log_values, on_epoch=True, on_step=True, logger=True, prog_bar=True)
        return {"loss": loss, "z": anc_word}

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, mode="valid")

    def configure_optimizers(self):
        if self.optim == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optim == "adamw":
            assert self.wt_decay > 0, "weight decay value must be non zero"
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wt_decay)
        else:
            raise NotImplementedError

        if self.lr_scheduler:
            scheduler = transformers.get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=10, num_training_steps=300
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
        else:
            return {"optimizer": optimizer}

    @staticmethod
    def _cosine_distance(x: torch.Tensor, y: torch.Tensor) -> float:
        assert x.dim() == y.dim() == 3, (
            f"the inputs must be 3 dimensional (batch, seq_len, feat_dims) tensor"
        )
        distance = 1 - torch.einsum("bqd, bkd -> bqk", [x, y])
        return distance

    @classmethod
    def load_from_pretrained(cls, checkpoint, gpu_index):
        model = cls.load_from_checkpoint(checkpoint, map_location=torch.device("cuda:" + str(gpu_index)))
        model.eval()
        return model
