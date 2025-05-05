import torch

from .f0_utils import cents_to_hz


class F0Selector(torch.nn.Module):
    def __init__(self, selection_strategy, f_min, f0_classes_cent, weighted_average_delta=None):
        super().__init__()

        assert selection_strategy in [
            "argmax",
            "parabolic_interpolation",
            "local_weighted_average",
        ], "'selection_strategy' must be one of 'argmax', 'parabolic_interpolation', or 'local_weighted_average'!"

        self.selection_strategy = selection_strategy
        self.f_min = f_min
        self.register_buffer("f0_classes_cent", f0_classes_cent)
        self.cent_step = f0_classes_cent[1]
        self.weighted_average_delta = weighted_average_delta

    def argmax(self, x_probs):
        max_idx = x_probs.argmax(dim=-1)
        f0_cent = self.f0_classes_cent[max_idx]
        f0_hz = cents_to_hz(f0_cent, f_ref=self.f_min)
        f0_hz[x_probs.sum(dim=-1) == 0] = 0
        return f0_hz

    @staticmethod
    def _parabolic_interpolation(y1, y2, y3):
        """
        Parabolic interpolation of an extremal value given three samples with equal spacing on the x-axis.
        The middle value y2 is assumed to be the extremal sample of the three.

        Adapted from: https://github.com/groupmm/libf0/blob/main/libf0/yin.py

        Parameters
        ----------
        y1: f(x1)
        y2: f(x2)
        y3: f(x3)

        Returns
        -------
        x_interp: Interpolated x-value (relative to x3-x2)
        y_interp: Interpolated y-value, f(x_interp)
        """
        a = (y1 + y3 - 2 * y2) / 2 + torch.finfo(torch.float32).eps
        b = (y3 - y1) / 2
        x_interp = -b / (2 * a)
        y_interp = y2 - (b**2) / (4 * a)

        return x_interp, y_interp

    def parabolic_interpolation(self, x_probs):
        max_idx = torch.argmax(x_probs, dim=-1, keepdim=True).type(torch.long)
        max_idx = torch.clamp(max_idx, min=1, max=x_probs.shape[-1] - 2)

        idx_shift, _ = self._parabolic_interpolation(
            torch.gather(x_probs, -1, max_idx - 1),
            torch.gather(x_probs, -1, max_idx),
            torch.gather(x_probs, -1, max_idx + 1),
        )

        f0_cent = (max_idx + idx_shift) * self.cent_step
        f0_hz = cents_to_hz(f0_cent, f_ref=self.f_min)
        return f0_hz

    def local_weighted_average(self, x_probs):
        leading_dims = x_probs.shape[:-1]
        n_bins = x_probs.shape[-1]

        max_idx = torch.argmax(x_probs, dim=-1).type(torch.long)

        start = torch.clamp(max_idx - self.weighted_average_delta, min=0)
        end = torch.clamp(max_idx + self.weighted_average_delta, max=n_bins - 1)

        indices = torch.arange(n_bins, device=x_probs.device).expand(*leading_dims, -1)
        mask = torch.logical_and(indices >= start.unsqueeze(-1), indices <= end.unsqueeze(-1))

        x_probs_masked = x_probs * mask

        weighted_average = torch.sum(x_probs_masked * self.f0_classes_cent, dim=-1)
        weight_sum = torch.sum(x_probs_masked, dim=-1)

        f0_cent = weighted_average / weight_sum
        f0_hz = cents_to_hz(f0_cent, f_ref=self.f_min)
        return f0_hz

    def forward(self, x_probs):
        if self.selection_strategy == "argmax":
            return self.argmax(x_probs)
        elif self.selection_strategy == "parabolic_interpolation":
            return self.parabolic_interpolation(x_probs)
        elif self.selection_strategy == "local_weighted_average":
            return self.local_weighted_average(x_probs)
