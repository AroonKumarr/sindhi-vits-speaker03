"""
VITS 4-Sample Overfitting Training Script for Kaggle (NVIDIA CUDA)
===================================================================
Self-contained runner for Kaggle Notebooks.
Uses local wavs/ and metadata.json.
"""

import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from models import (
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    SynthesizerTrn,
    discriminator_loss,
    duration_loss,
    feature_matching_loss,
    generator_adversarial_loss,
    kl_loss,
    mel_spectrogram_loss,
)

# ── Load Metadata ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
with open(ROOT / "metadata.json") as f:
    META = json.load(f)

SYMBOLS = META["symbols"]
_SYM2ID = {s: i for i, s in enumerate(SYMBOLS)}
_PAD_ID = _SYM2ID.get("<pad>", 0)


def text_to_sequence(text: str) -> List[int]:
    unk_id = _SYM2ID.get("<unk>", _SYM2ID.get("؟", 1))
    return [_SYM2ID.get(ch, unk_id) for ch in text]


# ── Collate & Dataset ─────────────────────────────────────────────────────────
class SindhiVITSCollate:
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        sample_ids = [b["sample_id"] for b in batch]
        raw_texts = [b["raw_text"] for b in batch]

        text_lengths = torch.LongTensor([len(b["text"]) for b in batch])
        max_text_len = int(torch.max(text_lengths).item())
        text_padded = torch.zeros((len(batch), max_text_len), dtype=torch.long).fill_(_PAD_ID)
        for i, b in enumerate(batch):
            text_padded[i, : len(b["text"])] = b["text"]

        spec_lengths = torch.LongTensor([b["spec"].size(-1) for b in batch])
        max_spec_len = int(torch.max(spec_lengths).item())
        spec_padded = torch.zeros((len(batch), 513, max_spec_len), dtype=torch.float32)
        for i, b in enumerate(batch):
            spec_padded[i, :, : b["spec"].size(-1)] = b["spec"]

        wav_lengths = torch.LongTensor([b["wav"].size(-1) for b in batch])
        max_wav_len = int(torch.max(wav_lengths).item())
        wav_padded = torch.zeros((len(batch), 1, max_wav_len), dtype=torch.float32)
        for i, b in enumerate(batch):
            wav_padded[i, :, : b["wav"].size(-1)] = b["wav"]

        return {
            "sample_ids": sample_ids,
            "raw_texts": raw_texts,
            "text_padded": text_padded,
            "text_lengths": text_lengths,
            "spec_padded": spec_padded,
            "spec_lengths": spec_lengths,
            "wav_padded": wav_padded,
            "wav_lengths": wav_lengths,
        }


class KaggleOverfitDataset(Dataset):
    def __init__(self, samples: List[Dict[str, str]], wav_dir: Path):
        self.samples = samples
        self.wav_dir = wav_dir

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        seq = torch.LongTensor(text_to_sequence(item["text"]))
        wav_path = self.wav_dir / item["wav_name"]
        data, sr = sf.read(str(wav_path))
        if data.ndim > 1:
            data = data.mean(axis=1)
        wav = torch.FloatTensor(data)
        hann = torch.hann_window(1024)
        spec = torch.abs(
            torch.stft(
                wav, 1024, 256, 1024, window=hann, center=False, return_complex=True
            )
        )
        return {
            "sample_id": item["sample_id"],
            "text": seq,
            "raw_text": item["text"],
            "spec": spec,
            "wav": wav.unsqueeze(0),
        }


# ── Training Helpers ──────────────────────────────────────────────────────────
def set_seed(seed: int = 20261117):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_reconstruction(net_g, dataset, device, out_dir: Path, tag: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    net_g.eval()
    with torch.no_grad():
        for item in dataset.samples:
            sid = item["sample_id"]
            wav_p = dataset.wav_dir / item["wav_name"]
            data, _ = sf.read(str(wav_p))
            if data.ndim > 1:
                data = data.mean(axis=1)
            wav = torch.FloatTensor(data).unsqueeze(0).to(device)
            hann = torch.hann_window(1024, device=device)
            spec = torch.abs(
                torch.stft(
                    wav.squeeze(0), 1024, 256, 1024, window=hann, center=False, return_complex=True
                )
            ).unsqueeze(0)
            spec_len = torch.tensor([spec.size(-1)], device=device)

            z, m_q, logs_q, y_mask = net_g.enc_q(spec, spec_len)
            z_p, _ = net_g.flow(z, y_mask)
            z_recon, _ = net_g.flow(z_p, y_mask, reverse=True)
            o = net_g.dec(z_recon * y_mask).squeeze().cpu().numpy()
            mx = np.abs(o).max()
            if mx > 0.95:
                o = o / mx * 0.95
            out_p = out_dir / f"{sid}_recon_{tag}.wav"
            sf.write(str(out_p), o, 16000)


def synthesize_all(net_g, dataset, device, out_dir: Path, step: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    net_g.eval()
    with torch.no_grad():
        for item in dataset.samples:
            sid = item["sample_id"]
            seq = torch.LongTensor(text_to_sequence(item["text"])).unsqueeze(0).to(device)
            seq_len = torch.tensor([seq.size(1)], device=device)
            audio = net_g.infer(seq, seq_len, noise_scale=0.667).squeeze().cpu().numpy()
            mx = np.abs(audio).max()
            if mx > 0.95:
                audio = audio / mx * 0.95
            out_p = out_dir / f"{sid}_step{step:04d}.wav"
            sf.write(str(out_p), audio, 16000)


def train(max_steps: int = 5000, gan_warmup_steps: int = 500):
    set_seed(20261117)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Running on device: {device}")
    if device.type == "cuda":
        print(f"   GPU Model: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    with open(ROOT / "overfit_4_samples.json") as f:
        cfg = json.load(f)

    dataset = KaggleOverfitDataset(META["samples"], ROOT / "wavs")
    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=SindhiVITSCollate())

    out_audio = ROOT / "output_audio"
    out_ckpts = ROOT / "checkpoints"
    out_audio.mkdir(parents=True, exist_ok=True)
    out_ckpts.mkdir(parents=True, exist_ok=True)

    mc = cfg["model"]
    net_g = SynthesizerTrn(
        n_vocab=len(SYMBOLS),
        spec_channels=513,
        segment_size=cfg["train"]["segment_size"],
        inter_channels=mc["inter_channels"],
        hidden_channels=mc["hidden_channels"],
        filter_channels=mc["filter_channels"],
        n_heads=mc["n_heads"],
        n_layers=mc["n_layers"],
        kernel_size=mc["kernel_size"],
        p_dropout=mc["p_dropout"],
        resblock_kernel_sizes=tuple(mc["resblock_kernel_sizes"]),
        resblock_dilation_sizes=tuple(tuple(x) for x in mc["resblock_dilation_sizes"]),
        upsample_rates=tuple(mc["upsample_rates"]),
        upsample_initial_channel=mc["upsample_initial_channel"],
        upsample_kernel_sizes=tuple(mc["upsample_kernel_sizes"]),
        n_flow_layers=mc["n_flow_layers"],
    ).to(device)

    net_d = nn.ModuleList([MultiPeriodDiscriminator(), MultiScaleDiscriminator()]).to(device)

    opt_g = torch.optim.AdamW(net_g.parameters(), lr=0.0002, betas=(0.8, 0.99), eps=1e-9)
    opt_d = torch.optim.AdamW(net_d.parameters(), lr=0.0002, betas=(0.8, 0.99), eps=1e-9)

    sched_g = torch.optim.lr_scheduler.ExponentialLR(opt_g, gamma=0.999875)
    sched_d = torch.optim.lr_scheduler.ExponentialLR(opt_d, gamma=0.999875)

    print("Initial baseline audio & reconstruction test...")
    synthesize_all(net_g, dataset, device, out_audio, step=0)
    run_reconstruction(net_g, dataset, device, out_audio, tag="init")

    step = 0
    mel_ema = 0.6
    t0 = time.time()

    print(f"\nTraining for {max_steps} steps (Warm-up: first {gan_warmup_steps} steps)...")
    batch = next(iter(loader))

    while step < max_steps:
        step += 1
        is_warmup = step <= gan_warmup_steps

        # Move to device
        x = batch["text_padded"].to(device)
        x_len = batch["text_lengths"].to(device)
        spec = batch["spec_padded"].to(device)
        spec_len = batch["spec_lengths"].to(device)
        wav = batch["wav_padded"].to(device)

        net_g.train()
        (
            y_hat,
            l_length,
            attn,
            ids_slice,
            x_mask,
            z_mask,
            (z, z_p, m_p, logs_p, m_q, logs_q),
        ) = net_g(x, x_len, spec, spec_len)

        # Slice real wav
        seg_size = cfg["train"]["segment_size"]
        wav_slices = []
        for b_idx in range(wav.size(0)):
            s_idx = ids_slice[b_idx].item()
            wav_slices.append(wav[b_idx : b_idx + 1, :, s_idx : s_idx + seg_size])
        wav_slice = torch.cat(wav_slices, dim=0)

        # Mel Loss
        loss_mel = mel_spectrogram_loss(wav_slice, y_hat)
        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask)
        loss_dur = duration_loss(l_length, attn.detach(), x_mask)

        mel_raw = loss_mel.item()
        mel_ema = 0.95 * mel_ema + 0.05 * mel_raw

        if is_warmup:
            loss_g = 45.0 * loss_mel + 1.0 * loss_kl + 1.0 * loss_dur
            opt_g.zero_grad()
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(net_g.parameters(), 1.0)
            opt_g.step()
            loss_d_val = 0.0
        else:
            # Full GAN
            net_d.train()
            y_d_hat_r, y_d_hat_g, _, _ = net_d[0](wav_slice, y_hat.detach())
            loss_d_p, _, _ = discriminator_loss(y_d_hat_r, y_d_hat_g)
            y_d_hat_r_s, y_d_hat_g_s, _, _ = net_d[1](wav_slice, y_hat.detach())
            loss_d_s, _, _ = discriminator_loss(y_d_hat_r_s, y_d_hat_g_s)
            loss_d = loss_d_p + loss_d_s

            opt_d.zero_grad()
            loss_d.backward()
            torch.nn.utils.clip_grad_norm_(net_d.parameters(), 1.0)
            opt_d.step()
            loss_d_val = loss_d.item()

            y_d_hat_r, y_d_hat_g, fmap_r_p, fmap_g_p = net_d[0](wav_slice, y_hat)
            loss_g_p, _ = generator_adversarial_loss(y_d_hat_g)
            loss_fm_p = feature_matching_loss(fmap_r_p, fmap_g_p)

            y_d_hat_r_s, y_d_hat_g_s, fmap_r_s, fmap_g_s = net_d[1](wav_slice, y_hat)
            loss_g_s, _ = generator_adversarial_loss(y_d_hat_g_s)
            loss_fm_s = feature_matching_loss(fmap_r_s, fmap_g_s)

            loss_g = (
                45.0 * loss_mel
                + 1.0 * loss_kl
                + 1.0 * loss_dur
                + (loss_g_p + loss_g_s)
                + 2.0 * (loss_fm_p + loss_fm_s)
            )

            opt_g.zero_grad()
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(net_g.parameters(), 1.0)
            opt_g.step()

        sched_g.step()
        sched_d.step()

        if step % 25 == 0 or step == 1:
            phase = "WARMUP" if is_warmup else "GAN"
            elapsed = time.time() - t0
            rate = step / elapsed
            eta_min = (max_steps - step) / rate / 60
            print(
                f"Step [{step:4d}/{max_steps}] [{phase:6s}] | "
                f"mel_raw: {mel_raw:.4f}  mel_EMA: {mel_ema:.4f}  "
                f"kl: {loss_kl.item():.4f}  dur: {loss_dur.item():.4f}  "
                f"g: {loss_g.item():.2f}  d: {loss_d_val:.4f}  "
                f"ETA: {eta_min:.1f}min ({rate:.2f} it/s)"
            )

        if step in [100, 500, 1000, 2000, 3000, 5000]:
            print(f"💾 Checkpoint @ step {step}...")
            synthesize_all(net_g, dataset, device, out_audio, step=step)
            run_reconstruction(net_g, dataset, device, out_audio, tag=f"step{step:04d}")
            torch.save(net_g.state_dict(), out_ckpts / f"G_v2_step{step:04d}.pth")

    print("\n✅ Training Complete!")


if __name__ == "__main__":
    train()
