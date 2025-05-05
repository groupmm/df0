import math
import torch
import torch.nn.functional as F

from .f0_utils import get_log_frequencies
from .f0_selection import F0Selector


class dYIN(torch.nn.Module):
    """
    This module implements a differentiable version of the YIN algorithm for F0 estimation, dubbed dYIN.

    Parameters:
        fs (int): Sampling frequency in Hz
        frame_size (int): Frame size in samples
        hop_size (int): Hop size in samples
        f0_min (float): Lowest detectable F0 in Hz
        f0_max (float): Highest detectable F0 in Hz
        f0_r_cent (float): Output resolution in cents
        f0_selection_strategy (str): Specifies the F0 selection strategy.
            Options: [None, "argmax", "parabolic_interpolation", or "local_weighted_average"]
    """

    def __init__(
        self,
        fs=16000,
        frame_size=512,
        hop_size=320,
        f0_min=55.0,
        f0_max=3520.0,
        f0_r_cent=10.0,
        f0_selection_strategy=None,
    ):
        super().__init__()

        self.fs = fs
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.f0_r_cent = f0_r_cent
        self.frame_size = frame_size
        self.hop_size = hop_size

        self.tau_max = math.floor(fs / f0_min) + 1
        assert self.tau_max < self.frame_size, "Frame size too small for chosen 'f0_min'!"

        # calculate log-frequency axis and corresponding time lags
        f0_classes_cent = get_log_frequencies(
            f_min=f0_min, f_max=f0_max, cent_step=f0_r_cent, return_as="cent"
        )

        self.f0_classes_hz = get_log_frequencies(
            f_min=f0_min, f_max=f0_max, cent_step=f0_r_cent, return_as="hz"
        )

        time_positions = self.fs / self.f0_classes_hz

        # round time positions and calculate difference for parabolic interpolation
        self.register_buffer(
            "time_positions_rounded", torch.round(time_positions).type(torch.long)
        )
        self.register_buffer(
            "time_positions_difference", time_positions - self.time_positions_rounded
        )

        if f0_selection_strategy is not None:
            self.f0_selector = F0Selector(
                selection_strategy=f0_selection_strategy,
                f_min=f0_min,
                f0_classes_cent=f0_classes_cent,
                weighted_average_delta=80 // self.f0_r_cent,
            )
        else:
            self.f0_selector = None

    def compute_batched_autocorrelation(self, signal, dim=-1):
        """
        Method for computing the autocorrelation function for a batch of signals.
        """
        # convolution via FFT
        fft_size = 2 ** (
            math.ceil(
                torch.log(torch.as_tensor(2 * signal.shape[dim] - 1))
                / torch.log(torch.as_tensor(2))
            )
            # int(torch.log(torch.tensor(signal.shape[dim])) // torch.log(torch.tensor([2]))) + 1
        )
        fft = torch.fft.rfft(signal, fft_size, dim=dim)
        auto_corr = torch.fft.irfft(fft * fft.conj(), dim=dim)

        return auto_corr

    def compute_cmndf(self, signal):
        """
        Method for computing the cumulative mean normalized difference function (CMNDF).
        """
        corr = self.compute_batched_autocorrelation(signal)[..., : self.tau_max + 1]

        # difference function
        energy_acc = torch.cumsum(signal**2, axis=-1)
        energy_loc = energy_acc[..., -(self.tau_max + 1) :] - energy_acc[..., : (self.tau_max + 1)]
        diff = (
            energy_loc[..., [0]] + energy_loc[..., : self.tau_max + 1]
        ) - 2 * corr  # first value corresponds to tau=0

        # ensure non-negativity
        diff = diff - diff.min(dim=-1, keepdim=True).values

        # tau > 0
        cmndf = (
            diff[..., 1:]
            * torch.arange(1, diff.shape[-1], device=diff.device)
            / torch.maximum(
                diff[..., 1:].cumsum(dim=-1), torch.as_tensor(torch.finfo(torch.float32).eps)
            )
        )

        # tau = 0
        cmndf = torch.cat([torch.ones(*cmndf.shape[:-1], 1), cmndf], dim=-1)

        return cmndf  # first value corresponds to tau=0

    def cmndf_to_logits_and_probs(self, cmndf):
        """
        Method for converting the cumulative mean normalized difference function (CMNDF)
        into a probability mass function defined over a log-frequency axis.
        Applies parabolic interpolation for resampling.
        """
        # parabolic interpolation
        cmndf_padded = torch.cat([cmndf[..., [0]], cmndf, cmndf[..., [-1]]], dim=-1)

        a = 0.5 * cmndf_padded[..., :-2] - cmndf + 0.5 * cmndf_padded[..., 2:]
        b = -0.5 * cmndf_padded[..., :-2] + 0.5 * cmndf_padded[..., 2:]
        c = cmndf_padded

        logits = (
            a[..., self.time_positions_rounded] * self.time_positions_difference**2
            + b[..., self.time_positions_rounded] * self.time_positions_difference
            + c[..., self.time_positions_rounded]
        )

        # normalization
        logits = logits * (-1)
        probs = F.softmax(logits, dim=-1)

        return logits, probs

    def forward(self, x):
        # padding for centered estimates
        x_pad = F.pad(x, (self.frame_size // 2, self.frame_size // 2 - 1))

        # cut out frames
        x_frames = x_pad.unfold(-1, self.frame_size, self.hop_size)

        # compute CMNDF
        cmndf = self.compute_cmndf(x_frames)

        # evaluate CMNDF at right positions using parabolic interpolation
        logits, probs = self.cmndf_to_logits_and_probs(cmndf)

        out = {
            "logits": logits,
            "probs": probs,
        }

        # F0 selection
        if self.f0_selector is not None:
            out["f0_hz"] = self.f0_selector(probs)

        return out
