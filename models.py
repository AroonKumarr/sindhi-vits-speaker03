"""
Complete VITS Architecture Implementation for Sindhi Single-Speaker TTS.

Includes:
- TextEncoder (Prior Encoder)
- PosteriorEncoder (Spectrogram Encoder)
- ResidualCouplingBlock (Normalizing Flow - Forward & Reverse)
- MonotonicAlignmentSearch (MAS)
- DurationPredictor
- Generator (HiFi-GAN with Multi-Receptive Field Fusion)
- MultiPeriodDiscriminator & MultiScaleDiscriminator
- Complete Loss Functions (Mel, KL, Feature Matching, Adversarial, Duration)
- SynthesizerTrn (End-to-End VITS Training & Inference)

Author: Senior TTS Research Engineer & PyTorch Developer
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import remove_weight_norm, spectral_norm, weight_norm


# ---------------------------------------------------------------------------
# 1. Monotonic Alignment Search (MAS) - Pure Vectorized Implementation
# ---------------------------------------------------------------------------
def maximum_path(
    value: torch.Tensor,
    mask: torch.Tensor,
    x_lengths: Optional[torch.Tensor] = None,
    y_lengths: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Monotonic Alignment Search dynamic programming.
    value: [B, T_text, T_spec] log likelihood matrix
    mask:  [B, T_text, T_spec] binary mask
    Returns: [B, T_text, T_spec] binary alignment path
    """
    device = value.device
    dtype = value.dtype
    b, t_t, t_s = value.shape

    path = torch.zeros_like(value)
    for b_idx in range(b):
        v = value[b_idx].detach().cpu().numpy()
        m = mask[b_idx].detach().cpu().numpy()
        p = np.zeros_like(v)

        # Actual valid lengths for this utterance
        if x_lengths is not None and y_lengths is not None:
            actual_t = int(x_lengths[b_idx].item())
            actual_s = int(y_lengths[b_idx].item())
        else:
            actual_t = int(m[:, 0].sum()) if m[:, 0].sum() > 0 else t_t
            actual_s = int(m[0, :].sum()) if m[0, :].sum() > 0 else t_s

        actual_t = min(actual_t, t_t)
        actual_s = min(actual_s, t_s)

        dp = np.full((actual_t, actual_s), -np.inf, dtype=np.float32)
        dp[0, 0] = v[0, 0]

        # Vectorized dynamic programming across text dimension
        for t in range(1, actual_s):
            prev_same = dp[:, t - 1]
            prev_diag = np.pad(dp[:-1, t - 1], (1, 0), constant_values=-np.inf)
            dp[:, t] = np.maximum(prev_same, prev_diag) + v[:actual_t, t]

        # Monotonic Traceback from end of actual sequence
        curr_text = actual_t - 1
        for t in range(actual_s - 1, -1, -1):
            p[curr_text, t] = 1.0
            if curr_text > 0 and t > 0:
                if dp[curr_text - 1, t - 1] >= dp[curr_text, t - 1]:
                    curr_text -= 1

        path[b_idx] = torch.from_numpy(p).to(device=device, dtype=dtype)
    return path


# ---------------------------------------------------------------------------
# 2. Sequence Utility Functions & Masks
# ---------------------------------------------------------------------------
def sequence_mask(length: torch.Tensor, max_length: Optional[int] = None) -> torch.Tensor:
    if max_length is None:
        max_length = int(torch.max(length).item())
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def generate_path(duration: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Generate alignment path from integer durations."""
    b, t_t = duration.shape
    if mask is not None:
        t_s = mask.size(-1)
    else:
        t_s = int(torch.max(torch.sum(duration, dim=1)).item())
    path = torch.zeros(b, t_t, t_s, dtype=torch.float32, device=duration.device)
    for i in range(b):
        cur_t = 0
        for j in range(t_t):
            d = int(duration[i, j].item())
            if d > 0 and cur_t < t_s:
                end_t = min(cur_t + d, t_s)
                path[i, j, cur_t:end_t] = 1.0
                cur_t = end_t
    return path


def rand_slice_segments(
    x: torch.Tensor, x_lengths: torch.Tensor, segment_size: int = 4
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Slice random contiguous segments for waveform generator training."""
    b, d, t = x.size()
    ids_str = torch.zeros(b, dtype=torch.long, device=x.device)
    output = torch.zeros(b, d, segment_size, dtype=x.dtype, device=x.device)
    for i in range(b):
        max_idx = int(x_lengths[i].item()) - segment_size
        if max_idx > 0:
            idx = torch.randint(0, max_idx + 1, (1,), device=x.device).item()
        else:
            idx = 0
        ids_str[i] = idx
        output[i] = x[i, :, idx : idx + segment_size]
    return output, ids_str


# ---------------------------------------------------------------------------
# 3. Text Encoder (Prior Encoder)
# ---------------------------------------------------------------------------
class TextEncoder(nn.Module):
    def __init__(
        self,
        n_vocab: int,
        out_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

        self.prenet = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(p_dropout),
        )

        encoder_layers = []
        for _ in range(n_layers):
            encoder_layers.append(
                nn.Sequential(
                    nn.Conv1d(hidden_channels, filter_channels, kernel_size, padding=kernel_size // 2),
                    nn.ReLU(),
                    nn.Dropout(p_dropout),
                    nn.Conv1d(filter_channels, hidden_channels, 1),
                    nn.Dropout(p_dropout),
                )
            )
        self.encoder_layers = nn.ModuleList(encoder_layers)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(
        self, x: torch.Tensor, x_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: [B, T_text] token IDs
        Returns: x (hidden), m (prior mean), logs (prior log_std), mask
        """
        h = self.emb(x).transpose(1, 2) * math.sqrt(self.hidden_channels)
        x_mask = sequence_mask(x_lengths, x.size(1)).unsqueeze(1).to(h.dtype)
        h = self.prenet(h) * x_mask

        for layer in self.encoder_layers:
            h = (h + layer(h)) * x_mask

        stats = self.proj(h) * x_mask
        m, logs = torch.chunk(stats, 2, dim=1)
        logs = torch.clamp(logs, min=-10.0, max=10.0)
        return h, m, logs, x_mask


# ---------------------------------------------------------------------------
# 4. Posterior Encoder (Spectrogram Encoder)
# ---------------------------------------------------------------------------
class PosteriorEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        kernel_size: int = 5,
        dilation_rate: int = 1,
        n_layers: int = 16,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = nn.ModuleList()
        for i in range(n_layers):
            dilation = dilation_rate**i
            self.enc.append(
                nn.Sequential(
                    nn.Conv1d(
                        hidden_channels,
                        hidden_channels,
                        kernel_size,
                        padding=(kernel_size - 1) * dilation // 2,
                        dilation=dilation,
                    ),
                    nn.ReLU(),
                )
            )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(
        self, y: torch.Tensor, y_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        y: [B, in_channels, T_spec] linear spectrogram
        Returns: z (sampled latent), m (mean), logs (log_std), mask
        """
        y_mask = sequence_mask(y_lengths, y.size(2)).unsqueeze(1).to(y.dtype)
        h = self.pre(y) * y_mask
        for layer in self.enc:
            h = (h + layer(h)) * y_mask
        stats = self.proj(h) * y_mask
        m, logs = torch.chunk(stats, 2, dim=1)
        logs = torch.clamp(logs, min=-10.0, max=10.0)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * y_mask
        return z, m, logs, y_mask


# ---------------------------------------------------------------------------
# 5. Normalizing Flow (Residual Coupling Block)
# ---------------------------------------------------------------------------
class ResidualCouplingLayer(nn.Module):
    def __init__(self, channels: int, hidden_channels: int, kernel_size: int = 5, n_layers: int = 4):
        super().__init__()
        self.half_channels = channels // 2
        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
            )
            for _ in range(n_layers)
        ])
        self.post = nn.Conv1d(hidden_channels, self.half_channels * 2, 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        x0, x1 = torch.chunk(x, 2, dim=1)
        h = self.pre(x0) * x_mask
        for layer in self.enc:
            h = (h + layer(h)) * x_mask
        stats = self.post(h) * x_mask
        m, logs = torch.chunk(stats, 2, dim=1)
        logs = torch.clamp(logs, min=-10.0, max=10.0)

        if not reverse:
            x1 = (m + x1 * torch.exp(logs)) * x_mask
            x = torch.cat([x0, x1], dim=1)
            logdet = torch.sum(logs, dim=[1, 2])
            return x, logdet
        else:
            x1 = ((x1 - m) * torch.exp(-logs)) * x_mask
            x = torch.cat([x0, x1], dim=1)
            return x, torch.zeros(x.size(0), device=x.device)


class ResidualCouplingBlock(nn.Module):
    def __init__(self, channels: int, hidden_channels: int, n_layers: int = 4):
        super().__init__()
        self.layers = nn.ModuleList([
            ResidualCouplingLayer(channels, hidden_channels) for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        total_logdet = torch.zeros(x.size(0), device=x.device)
        layers = reversed(self.layers) if reverse else self.layers
        for layer in layers:
            x, logdet = layer(x, x_mask, reverse=reverse)
            total_logdet = total_logdet + logdet
            x = torch.flip(x, dims=[1])  # Reversal between coupling layers
        return x, total_logdet


# ---------------------------------------------------------------------------
# 6. Duration Predictor
# ---------------------------------------------------------------------------
class DurationPredictor(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3, p_dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=kernel_size // 2)
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.drop1 = nn.Dropout(p_dropout)
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2)
        self.norm2 = nn.LayerNorm(hidden_channels)
        self.drop2 = nn.Dropout(p_dropout)
        self.proj = nn.Conv1d(hidden_channels, 1, 1)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        # x: [B, in_channels, T_text]
        h = self.conv1(x * x_mask).transpose(1, 2)
        h = self.drop1(F.relu(self.norm1(h))).transpose(1, 2) * x_mask
        h = self.conv2(h).transpose(1, 2)
        h = self.drop2(F.relu(self.norm2(h))).transpose(1, 2) * x_mask
        log_dur = self.proj(h) * x_mask
        return log_dur.squeeze(1)


# ---------------------------------------------------------------------------
# 7. Waveform Generator (HiFi-GAN)
# ---------------------------------------------------------------------------
class ResBlock1(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: Tuple[int, ...] = (1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=d, padding=(kernel_size - 1) * d // 2))
            for d in dilation
        ])
        self.convs2 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=(kernel_size - 1) // 2))
            for _ in dilation
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = xt + x
        return x


class Generator(nn.Module):
    def __init__(
        self,
        initial_channel: int = 192,
        resblock_kernel_sizes: Tuple[int, ...] = (3, 7, 11),
        resblock_dilation_sizes: Tuple[Tuple[int, ...], ...] = ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        upsample_rates: Tuple[int, ...] = (8, 8, 2, 2),
        upsample_initial_channel: int = 512,
        upsample_kernel_sizes: Tuple[int, ...] = (16, 16, 4, 4),
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = weight_norm(nn.Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3))

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            ch_in = upsample_initial_channel // (2**i)
            ch_out = upsample_initial_channel // (2 ** (i + 1))
            self.ups.append(
                weight_norm(nn.ConvTranspose1d(ch_in, ch_out, k, u, padding=(k - u) // 2))
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlock1(ch, k, d))

        self.conv_post = weight_norm(nn.Conv1d(ch_out, 1, 7, 1, padding=3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, initial_channel, T_slice]
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, 0.1)
            x = self.ups[i](x)
            xs = 0
            for j in range(self.num_kernels):
                xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x


# ---------------------------------------------------------------------------
# 8. Discriminators (Multi-Period & Multi-Scale)
# ---------------------------------------------------------------------------
class DiscriminatorP(nn.Module):
    def __init__(self, period: int, kernel_size: int = 5, stride: int = 3):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
            weight_norm(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
            weight_norm(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
            weight_norm(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
            weight_norm(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(kernel_size // 2, 0))),
        ])
        self.conv_post = weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(2),
            DiscriminatorP(3),
            DiscriminatorP(5),
            DiscriminatorP(7),
            DiscriminatorP(11),
        ])

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[torch.Tensor]], List[List[torch.Tensor]]]:
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(nn.Module):
    def __init__(self, use_spectral_norm: bool = False):
        super().__init__()
        norm_f = spectral_norm if use_spectral_norm else weight_norm
        self.convs = nn.ModuleList([
            norm_f(nn.Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True),
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        self.meanpools = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2),
        ])

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[torch.Tensor]], List[List[torch.Tensor]]]:
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


# ---------------------------------------------------------------------------
# 9. VITS Master Synthesizer (SynthesizerTrn)
# ---------------------------------------------------------------------------
class SynthesizerTrn(nn.Module):
    def __init__(
        self,
        n_vocab: int,
        spec_channels: int = 513,
        segment_size: int = 8192,
        inter_channels: int = 192,
        hidden_channels: int = 192,
        filter_channels: int = 768,
        n_heads: int = 2,
        n_layers: int = 6,
        kernel_size: int = 3,
        p_dropout: float = 0.1,
        resblock_kernel_sizes: Tuple[int, ...] = (3, 7, 11),
        resblock_dilation_sizes: Tuple[Tuple[int, ...], ...] = ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        upsample_rates: Tuple[int, ...] = (8, 8, 2, 2),
        upsample_initial_channel: int = 512,
        upsample_kernel_sizes: Tuple[int, ...] = (16, 16, 4, 4),
        n_flow_layers: int = 4,
    ):
        super().__init__()
        self.n_vocab = n_vocab
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.segment_size = segment_size

        self.enc_p = TextEncoder(
            n_vocab, inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout
        )
        self.enc_q = PosteriorEncoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16)
        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, n_flow_layers)
        self.dp = DurationPredictor(hidden_channels, 256, 3, 0.1)
        self.dec = Generator(
            inter_channels,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
        )

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        y_spec: torch.Tensor,
        y_lengths: torch.Tensor,
        y_wav: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Full VITS Training Step Forward Pass.
        """
        # 1. Prior Text Encoder
        h_text, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)

        # 2. Posterior Spectrogram Encoder
        z, m_q, logs_q, y_mask = self.enc_q(y_spec, y_lengths)

        # 3. Normalizing Flow (Forward: z -> z_0)
        z_p, logdet = self.flow(z, y_mask, reverse=False)

        # 4. Monotonic Alignment Search (MAS)
        with torch.no_grad():
            # Negative log likelihood matching
            # [B, inter_channels, T_spec] vs [B, inter_channels, T_text]
            s_p_sq_r = torch.exp(-2 * logs_p)
            neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True)
            neg_cent2 = torch.matmul(-0.5 * (z_p**2).transpose(1, 2), s_p_sq_r)
            neg_cent3 = torch.matmul(z_p.transpose(1, 2), (m_p * s_p_sq_r))
            neg_cent4 = torch.sum(-0.5 * (m_p**2) * s_p_sq_r, [1], keepdim=True)
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4
            neg_cent = neg_cent.transpose(1, 2)  # [B, T_text, T_spec]

            align_mask = torch.matmul(x_mask.transpose(1, 2), y_mask)  # [B, T_text, T_spec]
            attn = maximum_path(neg_cent, align_mask, x_lengths, y_lengths)  # [B, T_text, T_spec]

        # Target durations from alignment
        w = torch.sum(attn, dim=2)  # [B, T_text]

        # 5. Duration Predictor Forward Pass
        log_w_pred = self.dp(h_text.detach(), x_mask)

        # 6. Align Prior to Spectrogram dimension for KL loss
        m_p_aligned = torch.matmul(attn.transpose(1, 2), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p_aligned = torch.matmul(attn.transpose(1, 2), logs_p.transpose(1, 2)).transpose(1, 2)

        # 7. Sliced Waveform Generation
        z_slice, ids_slice = rand_slice_segments(z, y_lengths, segment_size=self.segment_size // 256)
        o_slice = self.dec(z_slice)

        # Slice corresponding ground-truth audio
        wav_slice = torch.zeros(y_wav.size(0), 1, self.segment_size, device=y_wav.device)
        for i in range(y_wav.size(0)):
            idx_wav = ids_slice[i] * 256
            wav_slice[i] = y_wav[i, :, idx_wav : idx_wav + self.segment_size]

        return {
            "o_slice": o_slice,
            "wav_slice": wav_slice,
            "ids_slice": ids_slice,
            "z": z,
            "z_p": z_p,
            "m_p": m_p_aligned,
            "logs_p": logs_p_aligned,
            "m_q": m_q,
            "logs_q": logs_q,
            "logdet": logdet,
            "y_mask": y_mask,
            "x_mask": x_mask,
            "attn": attn,
            "w_target": w,
            "log_w_pred": log_w_pred,
        }

    def infer(self, x: torch.Tensor, x_lengths: torch.Tensor, noise_scale: float = 0.667) -> torch.Tensor:
        """
        End-to-end Inference: Text -> Output Waveform.
        """
        # 1. Prior
        h_text, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)

        # 2. Predicted Duration
        log_w = self.dp(h_text, x_mask)
        w = torch.clamp(torch.round(torch.exp(log_w)), min=1) * x_mask.squeeze(1)

        # 3. Alignment path expansion
        y_lengths = torch.clamp_min(torch.sum(w, dim=1).long(), 1)
        y_mask = sequence_mask(y_lengths).unsqueeze(1).to(m_p.dtype)
        attn = generate_path(w, y_mask.squeeze(1))  # [B, T_text, T_spec]

        # 4. Expand prior distribution
        m_p_exp = torch.matmul(attn.transpose(1, 2), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p_exp = torch.matmul(attn.transpose(1, 2), logs_p.transpose(1, 2)).transpose(1, 2)

        # 5. Sample latent z_p and reverse Flow
        z_p = (m_p_exp + torch.randn_like(m_p_exp) * torch.exp(logs_p_exp) * noise_scale) * y_mask
        z, _ = self.flow(z_p, y_mask, reverse=True)

        # 6. Generator
        o = self.dec(z * y_mask)
        return o


# ---------------------------------------------------------------------------
# 10. Complete Loss Computations
# ---------------------------------------------------------------------------
def kl_loss(m_p: torch.Tensor, logs_p: torch.Tensor, m_q: torch.Tensor, logs_q: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """KL Divergence between two Gaussians: KL(q || p)."""
    kl = logs_p - logs_q - 0.5 + 0.5 * (torch.exp(2 * logs_q) + (m_q - m_p) ** 2) * torch.exp(-2 * logs_p)
    kl = torch.sum(kl * mask) / torch.sum(mask)
    return kl


def mel_spectrogram_loss(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """L1 Mel Spectrogram Loss."""
    # STFT parameters matching 16kHz
    hann_window = torch.hann_window(1024).to(dtype=y.dtype, device=y.device)
    spec_hat = torch.stft(y_hat.squeeze(1), 1024, 256, 1024, window=hann_window, return_complex=True)
    spec_gt = torch.stft(y.squeeze(1), 1024, 256, 1024, window=hann_window, return_complex=True)
    mag_hat = torch.abs(spec_hat)
    mag_gt = torch.abs(spec_gt)
    return F.l1_loss(mag_hat, mag_gt)


def discriminator_loss(disc_real_outputs: List[torch.Tensor], disc_gen_outputs: List[torch.Tensor]) -> torch.Tensor:
    """Least-squares adversarial discriminator loss."""
    loss = 0.0
    for dr, dg in zip(disc_real_outputs, disc_gen_outputs):
        r_loss = torch.mean((1.0 - dr) ** 2)
        g_loss = torch.mean(dg**2)
        loss += r_loss + g_loss
    return loss


def generator_adversarial_loss(disc_gen_outputs: List[torch.Tensor]) -> torch.Tensor:
    loss = 0.0
    for dg in disc_gen_outputs:
        loss += torch.mean((1.0 - dg) ** 2)
    return loss


def feature_matching_loss(fmap_r: List[List[torch.Tensor]], fmap_g: List[List[torch.Tensor]]) -> torch.Tensor:
    loss = 0.0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl.detach() - gl))
    return loss * 2.0


def duration_loss(log_w_pred: torch.Tensor, w_target: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
    log_w_target = torch.log(w_target + 1.0) * x_mask.squeeze(1)
    loss = F.mse_loss(log_w_pred * x_mask.squeeze(1), log_w_target, reduction="sum") / torch.sum(x_mask)
    return loss
