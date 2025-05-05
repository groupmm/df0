import torch
import torch.nn.functional as F

from .dswipe_utils import interp1d_linear, interp1d_linear_get_matrix, prime_and_one
from .f0_selection import F0Selector
from .f0_utils import get_log_frequencies, hz_to_cents


class dSWIPE(torch.nn.Module):
    """
    This module implements a differentiable version of the SWIPE algorithm for F0 estimation, dubbed dSWIPE.

    Implementation inspired by:
    https://github.com/groupmm/libf0/blob/main/libf0/swipe.py
    https://github.com/groupmm/libf0/blob/main/libf0/swipe_slim.py

    Parameters:
        fs (int): Sampling frequency in Hz
        hop_size (int): Hop size in samples
        erb_f_min (float): Lowest frequency of the intermediate ERB-based frequency axis in Hz
        erb_f_max (float): Highest frequency of the intermediate ERB-based frequency axis in Hz
        erb_r: Resolution of the intermediate ERB-based frequency axis in ERB units
        f0_min (float): Lowest detectable F0 in Hz
        f0_max (float): Highest detectable F0 in Hz
        f0_r_cent (float): Output resolution in cents
        template_params (dict): Parameter dictionary for the template comparison module
        f0_selection_strategy (str): Specifies the F0 selection strategy.
            Options: [None, "argmax", "parabolic_interpolation", or "local_weighted_average"]
    """

    def __init__(
        self,
        fs=16000,
        hop_size=320,
        erb_f_min=13.75,
        erb_f_max=8000,
        erb_r=0.1,
        f0_min=55.0,
        f0_max=3520.0,
        f0_r_cent=10,
        template_params={},
        f0_selection_strategy=None,
    ):

        super().__init__()

        self.fs = fs
        self.hop_size = hop_size

        # define set of ERB-based frequencies to which the spectrograms are resampled
        hz2erb = lambda hz: 21.4 * torch.log10(1 + torch.tensor(hz) / 229)
        erb2hz = lambda erbs: (10 ** (erbs / 21.4) - 1) * 229

        f_erb = torch.arange(
            hz2erb(erb_f_min),
            hz2erb(erb_f_max),
            erb_r,
        )
        self.f_erb_hz = erb2hz(f_erb)

        # define set of F0 classes
        f0_classes_hz = get_log_frequencies(f_min=f0_min, f_max=f0_max, cent_step=f0_r_cent)
        self.register_buffer("f0_classes_hz", f0_classes_hz)

        self.template_comparison = TemplateComparison(
            f_erb_hz=self.f_erb_hz,
            f0_classes_hz=self.f0_classes_hz,
            **template_params,
        )

        # determine set of power-of-two window sizes
        L_opt = torch.log2(self.fs * 8 / torch.tensor([f0_min, f0_max]))
        L_rnd = torch.arange(torch.floor(L_opt[1]), torch.ceil(L_opt[0]) + 1).type(torch.int)
        self.N_pow2 = 2**L_rnd

        # compute Hann windows for STFT computations
        for N in self.N_pow2:
            self.register_buffer(f"hann_window_{N}", torch.hann_window(N), persistent=False)

        # for every window size: precompute matrix to perform linear interpolation across the frequency axis; linear to ERB
        f_coef_lin_hz = lambda N: (torch.arange(N // 2 + 1) * self.fs / N)
        for N in self.N_pow2:
            interp1d_matrix = interp1d_linear_get_matrix(f_coef_lin_hz(N), self.f_erb_hz)
            self.register_buffer(f"erb_interp1d_matrix_{N}", interp1d_matrix, persistent=False)

        # compute combining weights for power-of-two window sizes to approximate optimal window sizes
        err = torch.abs(
            torch.log2(8 * self.fs / self.f0_classes_hz) - torch.log2(torch.max(self.N_pow2))
        ).unsqueeze(dim=1)

        octaves = torch.arange(self.N_pow2.numel()).flip(0).unsqueeze(dim=0)
        candidates = (err > octaves - 1) & (err < octaves + 1)
        mu = torch.zeros_like(candidates, dtype=torch.float32)
        mu[candidates] = 1 - torch.abs((err - octaves)[candidates])
        mu = mu.unsqueeze(dim=1).unsqueeze(dim=0)  # (1, n_f0_classes, 1, n_windows)
        self.register_buffer("combining_weights", mu)

        # F0 selection
        f0_classes_cent = hz_to_cents(f0_classes_hz, f_ref=f0_min)

        if f0_selection_strategy is not None:
            self.f0_selector = F0Selector(
                selection_strategy=f0_selection_strategy,
                f_min=f0_min,
                f0_classes_cent=f0_classes_cent,
                weighted_average_delta=80 // f0_r_cent,
            )
        else:
            self.f0_selector = None

    @property
    def templates(self):
        return self.template_comparison.templates

    def compute_erb_features(self, x):
        # define target time axis
        t_target = tuple((torch.arange(0, x.shape[-1], self.hop_size) / self.fs).tolist())

        # compute ERB-based spectra for all window sizes
        x_erb_all = []

        for i, N in enumerate(self.N_pow2):
            if self.training:
                x_pad = F.pad(x, (0, N))
                hop_size = N // 2  # minimize computational cost and redundancy
            else:
                x_pad = x
                hop_size = self.hop_size  # avoid temporal resampling

            # compute magnitude spectrogram
            x_stft = torch.abs(
                torch.stft(
                    x_pad,
                    n_fft=N,
                    hop_length=hop_size,
                    win_length=N,
                    window=getattr(self, f"hann_window_{N}"),
                    pad_mode="constant",
                    center=True,
                    return_complex=True,
                )
            )

            # resample to ERB-based frequency axis
            x_erb = torch.matmul(
                torch.transpose(x_stft, 1, -1),
                getattr(self, f"erb_interp1d_matrix_{N}"),
            ).transpose(1, -1)

            # resample to target time axis
            if self.training:
                # compute current time axis
                t_curr = tuple((torch.arange(0, x_erb.shape[-1]) * hop_size / self.fs).tolist())

                x_erb = interp1d_linear(
                    x_erb,
                    x_old=t_curr,
                    x_new=t_target,
                    dim=-1,
                )

            x_erb_all.append(x_erb)

        return torch.stack(x_erb_all, dim=-1)  # (bs, n_erbs, n_frames, n_windows)

    def forward(self, x):
        if x.ndim == 1:
            x = x.unsqueeze(dim=0)
            remove_batch_dim = True
        elif x.ndim == 2:
            remove_batch_dim = False
        else:
            raise ValueError("x has too many dimensions; needs to be one- or two-dimensional.")

        # get ERB-based spectral representations
        x_erb = self.compute_erb_features(x)  # (bs, n_erbs, n_frames, n_windows)

        # magnitude compression
        x_erb = torch.sqrt(x_erb)  # (bs, n_erbs, n_frames, n_windows)

        # normalize magnitudes
        x_erb = F.normalize(x_erb, p=2.0, dim=1, eps=1e-12)  # (bs, n_erbs, n_frames, n_windows)

        # correlate spectral representations with templates
        S = self.template_comparison(x_erb)  # (bs, n_f0_classes, n_frames, n_windows)

        # compute weighted sum of the predictions made using different window lengths
        S_sum = torch.multiply(self.combining_weights, S).sum(
            dim=-1
        )  # (bs, n_f0_classes, n_frames)
        Y = F.softmax(S_sum, dim=1)  # (bs, n_f0_classes, n_frames)

        S_sum = S_sum.permute(0, 2, 1)  # (bs, n_frames, n_f0_classes)
        Y = Y.permute(0, 2, 1)  # (bs, n_frames, n_f0_classes)

        # remove batch dim if necessary
        if remove_batch_dim:
            S_sum = S_sum.squeeze(dim=0)
            Y = Y.squeeze(dim=0)

        out = {
            "logits": S_sum,
            "probs": Y,
        }

        # F0 selection
        if self.f0_selector is not None:
            out["f0_hz"] = self.f0_selector(out["probs"])

        return out


class TemplateComparison(torch.nn.Module):
    """
    This module compares ERB-based input audio spectrograms to F0 templates.
    Optionally: The F0 templates can be defined as trainable parameters.

    Parameters:
        f_erb_hz (torch.Tensor): ERB-based frequency axis in Hz
        f0_classes_hz (torch.Tensor): Output F0 classes in Hz
        decay_factor (float): Harmonic decay factor
        normalization_type (str): How to normalize the templates (options: ["l2", "l2+"])
        training_mode (str): How to compute the templates (options: [None, "F", "P", "C"]
            Default: None; behaves like original SWIPE with no trainable parameters
        template_init (str): How to initialize the templates if trainable (options: ["swipe", "rand"])
            (only relevant for training modes "F", "P", and "C")
        learned_lobe_n (int): Number of trainable parameters in case of a learnable lobe
            (only relevant for training mode "C")
        prototype_n_oct_above (int): Number of octaves which the prototype should cover above the F0
            (only relevant for training mode "P")
        prototype_n_oct_above (int): Number of octaves which the prototype should cover below the F0
            (only relevant for training mode "P")
        prototype_bins_per_oct (int): Number of bins per octave for the prototype template
            (only relevant for training mode "P")
    """

    def __init__(
        self,
        f_erb_hz,
        f0_classes_hz,
        decay_factor=0.5,
        normalization_type="l2",
        training_mode=None,
        template_init="swipe",
        learned_lobe_n=50,
        prototype_n_oct_above=4,
        prototype_n_oct_below=3,
        prototype_bins_per_oct=60,
    ):
        super().__init__()

        self.register_buffer("f_erb_hz", f_erb_hz)
        self.register_buffer("f0_classes_hz", f0_classes_hz)
        self.n_spectrum = f_erb_hz.numel()
        self.n_candidates = f0_classes_hz.numel()

        self.training_mode = training_mode
        self.normalization_type = normalization_type

        assert template_init in [
            "swipe",
            "rand",
        ], "Check template_init parameter..."

        if training_mode is None:
            self.register_buffer("decay_factor", torch.as_tensor(decay_factor))
            self.register_buffer("templates", self.get_original_templates())

        elif training_mode == "F":
            if template_init == "swipe":
                self.register_buffer("decay_factor", torch.as_tensor(decay_factor))
                self.register_parameter(
                    "templates", torch.nn.Parameter(self.get_original_templates())
                )
            elif template_init == "rand":
                self.register_parameter(
                    "templates",
                    torch.nn.Parameter(
                        torch.rand(self.f0_classes_hz.numel(), self.f_erb_hz.numel())
                    ),
                )

        elif training_mode == "P":
            # compute number of trainable parameters
            self.prototype_n_params = (
                prototype_n_oct_above + prototype_n_oct_below
            ) * prototype_bins_per_oct

            # harmonic indices for which to learn a coefficient
            self.prototype_harmonic_idx = (
                1
                / (2**prototype_n_oct_below)
                * 2 ** (torch.arange(self.prototype_n_params) / prototype_bins_per_oct)
            )

            # prototype initialization
            if template_init == "swipe":
                learned_template = torch.zeros(self.prototype_n_params)
                n_harmonics = torch.ceil(
                    self.prototype_harmonic_idx[-1] / self.prototype_harmonic_idx[0]
                ).type(torch.int)
                prime_numbers = prime_and_one(upto=n_harmonics)

                # loop through all prime harmonics
                for p in prime_numbers:
                    a = torch.abs(
                        self.prototype_harmonic_idx - p
                    )  # normalized distance between harmonic and current pitch candidate
                    main_peak_bins = a < 0.25
                    valley_bins = torch.logical_and(0.25 < a, a < 0.75)

                    learned_template[main_peak_bins] = torch.cos(
                        torch.tensor(2 * torch.pi) * self.prototype_harmonic_idx[main_peak_bins]
                    )
                    learned_template[valley_bins] += (
                        torch.cos(
                            torch.tensor(2 * torch.pi) * self.prototype_harmonic_idx[valley_bins]
                        )
                        / 2
                    )

                decay = 1.0 / torch.pow(self.prototype_harmonic_idx, decay_factor)

                self.register_parameter("prototype", torch.nn.Parameter(learned_template * decay))

            elif template_init == "rand":
                self.register_parameter(
                    "prototype",
                    torch.nn.Parameter(2 * torch.rand(self.prototype_n_params) - 1),
                )

            self.register_buffer(
                "prototype_interpolation_matrix", self.get_prototype_interpolation_matrix()
            )

        elif training_mode == "C":
            self._learned_lobe_n = learned_lobe_n
            self._learned_lobe_t = (
                torch.arange(self._learned_lobe_n) / (self._learned_lobe_n - 1) * 0.5
            )

            if template_init == "swipe":
                self.register_parameter(
                    "_learned_lobe",
                    torch.nn.Parameter(torch.cos(2 * torch.pi * (self._learned_lobe_t - 0.25))),
                )
            elif template_init == "rand":
                self.register_parameter(
                    "_learned_lobe",
                    torch.nn.Parameter(torch.rand(self._learned_lobe_n)),
                )

            self.register_buffer("_lobe_upsampling_matrix", self.get_lobe_interpolation_matrix())

            self.register_parameter(
                "decay_factor", torch.nn.Parameter(torch.as_tensor(decay_factor))
            )

        else:
            raise ValueError(f"Training mode {training_mode} unknown...")

        self.update_templates()

    def get_lobe_interpolation_matrix(self):
        """
        This method returns a matrix to derive all templates from a learned lobe via matrix multiplication.
        """
        upsampling_matrix = torch.zeros(self.n_candidates, self.n_spectrum, self._learned_lobe_n)

        for i, f in enumerate(self.f0_classes_hz):
            n_harmonics = torch.ceil(self.f_erb_hz[-1] / f).type(torch.int)
            prime_numbers = prime_and_one(upto=n_harmonics)

            # ratio between all frequencies and template fundamental frequency
            ratio = self.f_erb_hz / f

            # loop through all prime harmonics
            for p in prime_numbers:
                a = torch.abs(
                    ratio - p
                )  # absolute distance between harmonic and current pitch candidate

                # determine bins that form peak/valley for that harmonic
                main_peak_bins = a <= 0.25
                valley_bins = torch.logical_and(a > 0.25, a < 0.75)

                # modify ratio values since lobe is not zero-centered, but centered at 0.25
                ratio_modified = ratio - p
                ratio_modified[main_peak_bins] += 0.25
                ratio_modified[valley_bins] = torch.remainder(
                    ratio_modified[valley_bins] - 0.25, 1
                )

                considered_bins = torch.logical_or(main_peak_bins, valley_bins)

                # find indices in lobe
                idx_ceil = torch.argmax(
                    (
                        ratio_modified.cpu().repeat(self._learned_lobe_n, 1)
                        < self._learned_lobe_t.repeat(self.n_spectrum, 1).transpose(1, 0)
                    ).float(),
                    dim=0,
                )
                idx_floor = idx_ceil - 1

                # determine weight factors for lobe values
                alpha = (ratio_modified - self._learned_lobe_t[idx_floor]) / (
                    self._learned_lobe_t[idx_ceil] - self._learned_lobe_t[idx_floor]
                )
                beta = 1 - alpha

                # set other values than the considered ones to zero, and change sign of valley bins
                alpha[~considered_bins], beta[~considered_bins] = 0, 0
                alpha[valley_bins] *= -0.5
                beta[valley_bins] *= -0.5

                upsampling_matrix[i, torch.arange(self.n_spectrum), idx_floor] += beta
                upsampling_matrix[i, torch.arange(self.n_spectrum), idx_ceil] += alpha

        return upsampling_matrix

    def get_prototype_interpolation_matrix(self):
        """
        This method returns a matrix to derive all templates from a prototype template via matrix multiplication.
        """
        upsampling_matrix = torch.zeros(
            self.n_candidates, self.n_spectrum, self.prototype_n_params
        )

        for i, f in enumerate(self.f0_classes_hz):
            idx_new = self.f_erb_hz / f

            idx_ceil = torch.argmax(
                (
                    idx_new.cpu().repeat(self.prototype_n_params, 1)
                    < self.prototype_harmonic_idx.repeat(self.n_spectrum, 1).transpose(1, 0)
                ).float(),
                dim=0,
            )
            idx_floor = idx_ceil - 1

            alpha = (idx_new - self.prototype_harmonic_idx[idx_floor]) / (
                self.prototype_harmonic_idx[idx_ceil] - self.prototype_harmonic_idx[idx_floor]
            )
            beta = 1 - alpha

            alpha[
                (idx_new < self.prototype_harmonic_idx[0])
                | (idx_new > self.prototype_harmonic_idx[-1])
            ] = 0
            beta[
                (idx_new < self.prototype_harmonic_idx[0])
                | (idx_new > self.prototype_harmonic_idx[-1])
            ] = 0

            upsampling_matrix[i, torch.arange(self.n_spectrum), idx_floor] += beta
            upsampling_matrix[i, torch.arange(self.n_spectrum), idx_ceil] += alpha

        return upsampling_matrix

    def get_base_template_f(self, f):
        k = torch.zeros(self.n_spectrum)
        n_harmonics = torch.ceil(self.f_erb_hz[-1] / f).type(torch.int)
        prime_numbers = prime_and_one(upto=n_harmonics)

        ratio = self.f_erb_hz / f

        # loop through all prime harmonics
        for p in prime_numbers:
            a = torch.abs(
                ratio - p
            )  # normalized distance between harmonic and current pitch candidate
            main_peak_bins = a < 0.25
            valley_bins = torch.logical_and(0.25 < a, a < 0.75)

            k[main_peak_bins] = torch.cos(torch.tensor(2 * torch.pi) * ratio[main_peak_bins])
            k[valley_bins] += torch.cos(torch.tensor(2 * torch.pi) * ratio[valley_bins]) / 2

        return k

    def get_original_templates(self):
        # construct all templates using the standard cosine lobe
        templates_base = torch.zeros((self.n_candidates, self.n_spectrum))
        for i, f in enumerate(self.f0_classes_hz):
            templates_base[i, :] = self.get_base_template_f(f)

        # apply decay
        templates = torch.multiply(templates_base, self.get_decay())

        # normalize templates
        templates_normalized = templates / self.get_normalization_factor(templates)

        return templates_normalized

    def get_decay(self):
        # compute harmonic decay
        harmonic = torch.outer(1 / self.f0_classes_hz, self.f_erb_hz)
        decay = 1.0 / torch.pow(harmonic, self.decay_factor)
        return decay

    def get_normalization_factor(self, templates, eps=1e-12):
        # modified l2 norm: normalize using l2 norm of positive components
        if self.normalization_type == "l2+":
            return (
                torch.linalg.vector_norm(templates[templates > 0], dim=-1, keepdim=True, ord=2)
                + torch.ones(templates.shape[0], 1, device=templates.device) * eps
            )

        # standard l2 norm
        elif self.normalization_type == "l2":
            return (
                torch.linalg.vector_norm(templates, dim=-1, keepdim=True, ord=2)
                + torch.ones(templates.shape[0], 1, device=templates.device) * eps
            )

        else:
            raise ValueError

    def update_templates(self):
        if self.training_mode in [None, "F"]:
            return

        if self.training_mode == "P":
            # derive templates from a learned prototype template via stretching (linear intepolation)
            templates = torch.matmul(self.prototype_interpolation_matrix, self.prototype)

        elif self.training_mode == "C":
            # construct all templates using the learned lobe (linear interpolation)
            templates_no_decay = torch.matmul(self._lobe_upsampling_matrix, self._learned_lobe)
            templates = torch.multiply(templates_no_decay, self.get_decay())

        # normalize templates
        templates_normalized = templates / self.get_normalization_factor(templates)

        # overwrite templates in buffer
        self.register_buffer("templates", templates_normalized)

    def forward(self, x_erb):
        self.update_templates()
        S = torch.einsum(
            "abcd,eb->aecd", x_erb, self.templates
        )  # (bs, n_f0_classes, n_frames, n_windows)
        return S
