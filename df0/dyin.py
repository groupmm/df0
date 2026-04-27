import math
import torch
import torch.nn.functional as F

from .f0_utils import get_log_frequencies
from .f0_selection import F0Selector


class dYIN(torch.nn.Module):
    """
    Differentiable YIN (dYIN) module for F0 estimation.

    Parameters:
        fs (int): Sampling frequency in Hz.
        frame_size (int): Frame size in samples.
        hop_size (int): Hop size in samples.
        f0_min (float): Lowest detectable F0 in Hz.
        f0_max (float): Highest detectable F0 in Hz.
        f0_r_cent (float): Output frequency resolution in cents.
        f0_selection_strategy (str | None): F0 selection strategy.
            Options: None, "argmax", "parabolic_interpolation", "local_weighted_average".
    """

    def __init__(
        self,
        fs: int = 16000,
        frame_size: int = 1600,
        hop_size: int = 320,
        f0_min: float = 55.0,
        f0_max: float = 3520.0,
        f0_r_cent: float = 10.0,
        f0_selection_strategy: str | None = "argmax",
    ):
        super().__init__()

        assert isinstance(fs, int), "'fs' must be an integer."
        assert isinstance(frame_size, int), "'frame_size' must be an integer."
        assert isinstance(hop_size, int), "'hop_size' must be an integer."

        self.fs = fs
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.f0_r_cent = f0_r_cent
        self.frame_size = frame_size
        self.hop_size = hop_size

        self.tau_max = math.floor(fs / f0_min) + 1
        assert 2 * self.tau_max < self.frame_size, "Frame size too small for chosen 'f0_min'!"

        # calculate log-frequency axis and corresponding time lags
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
            f0_classes_cent = get_log_frequencies(
                f_min=f0_min, f_max=f0_max, cent_step=f0_r_cent, return_as="cent"
            )

            self.f0_selector = F0Selector(
                selection_strategy=f0_selection_strategy,
                f_min=f0_min,
                f0_classes_cent=f0_classes_cent,
                weighted_average_delta=80 // self.f0_r_cent,
            )
        else:
            self.f0_selector = None

    def compute_cmndf(self, signal):
        """Computes the cumulative mean normalized difference function (CMNDF) using a centered, symmetric variant."""
        start = self.tau_max
        end = self.frame_size - self.tau_max

        # autocorrelation
        fft_size = 2 ** math.ceil(
            torch.log(torch.as_tensor(2 * signal.shape[-1] - 1)) / torch.log(torch.as_tensor(2))
        )
        b_input = F.pad(signal[..., start:end], (start, self.tau_max))
        a = torch.fft.rfft(signal, fft_size, dim=-1)
        b = torch.fft.rfft(b_input, fft_size, dim=-1)
        raw = torch.fft.irfft(a * b.conj(), fft_size, dim=-1)
        r_plus = raw[..., : self.tau_max + 1]
        r_minus = torch.cat([raw[..., :1], raw[..., -self.tau_max :].flip(-1)], dim=-1)

        # energy terms
        sqrcs = F.pad(torch.cumsum(signal**2, dim=-1), [1, 0])
        e_center = sqrcs[..., [end]] - sqrcs[..., [start]]
        e_plus = (
            sqrcs[..., end : end + self.tau_max + 1] - sqrcs[..., start : start + self.tau_max + 1]
        )
        e_minus = sqrcs[..., end - self.tau_max : end + 1].flip(-1) - sqrcs[..., : start + 1].flip(
            -1
        )

        # difference function
        # diff = e_center + e_plus - 2 * r_plus                        # closer to original formulation but asymmetric
        diff = e_center + 0.5 * (e_plus + e_minus) - r_plus - r_minus  # symmetric variant

        diff = diff.clamp(min=0)  # numerical issues can yield small negative values

        # tau > 0
        cmndf = (
            diff[..., 1:]
            * torch.arange(1, diff.shape[-1], device=diff.device)
            / torch.maximum(
                diff[..., 1:].cumsum(dim=-1), torch.as_tensor(torch.finfo(torch.float32).eps)
            )
        )

        # tau = 0
        cmndf = torch.cat([torch.ones(*cmndf.shape[:-1], 1, device=cmndf.device), cmndf], dim=-1)

        return cmndf  # first value corresponds to tau=0

    def cmndf_to_logits_and_probs(self, cmndf):
        """Converts CMNDF to logits and probabilities over a log-frequency axis using parabolic interpolation."""
        # parabolic interpolation
        cmndf_padded = torch.cat([cmndf[..., [0]], cmndf, cmndf[..., [-1]]], dim=-1)

        a = 0.5 * cmndf_padded[..., :-2] - cmndf + 0.5 * cmndf_padded[..., 2:]
        b = -0.5 * cmndf_padded[..., :-2] + 0.5 * cmndf_padded[..., 2:]
        c = cmndf

        logits = (
            a[..., self.time_positions_rounded] * self.time_positions_difference**2
            + b[..., self.time_positions_rounded] * self.time_positions_difference
            + c[..., self.time_positions_rounded]
        )

        # normalization
        logits = logits * (-1)
        probs = F.softmax(logits, dim=-1)

        return logits, probs

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): Input waveform of shape (..., signal_length).

        Returns:
            dict with keys 'logits', 'probs', and optionally 'f0_hz'.
        """
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
