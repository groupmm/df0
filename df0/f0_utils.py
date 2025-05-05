import torch


def midi_to_hz(f_midi):
    """Converts MIDI pitches to frequencies in Hz

    Args:
        f_midi (float or torch.Tensor): MIDI pitches

    Returns:
        f_hz (torch.Tensor): Frequency value(s) in Hz
    """
    f_hz = 440 * 2 ** ((torch.as_tensor(f_midi) - 69) / 12)
    return f_hz


def hz_to_cents(f_hz, f_ref=55.0):
    """Converts frequencies in Hz to cents

    Args:
        f_hz (float or torch.Tensor): Frequency value(s) in Hz
        f_ref (float): Reference frequency in Hz (Default value = 55.0)

    Returns:
        f_cent (torch.Tensor): Frequency value(s) in cents
    """
    f_cent = 1200 * torch.log2(torch.as_tensor(f_hz) / f_ref)
    return f_cent


def cents_to_hz(f_cent, f_ref=55.0):
    """Converts frequencies in cents to Hz

    Args:
        f_cent (float or torch.Tensor): Frequency value(s) in cents
        f_ref (float): Reference frequency in Hz (Default value = 55.0)

    Returns:
        f_hz (torch.Tensor): Frequency value(s) in Hz
    """
    f_hz = f_ref * 2 ** (f_cent / 1200)
    return f_hz


def get_log_frequencies(f_min=55.0, f_max=1760.0, cent_step=10.0, n_freq=None, return_as="hz"):
    """Creates a discrete logarithmic frequency axis

    Args:
        f_min (float): Minimum frequency in Hz
        f_max (float): Maximum frequency in Hz
        cent_step (float): Distance between adjacent frequencies in cents
        n_freq (int): Number of frequency bins (optional, either 'f_max' or 'n_freq' must be None)
        return_as (str): Specifies whether the output is in Hz ('hz') or in cents ('cent')

    Returns:
        f_hz (torch.Tensor): Frequency value(s) in Hz
    """
    assert (f_max is None) ^ (n_freq is None), "Either f_max or n_freq must be None!"
    assert return_as in ["hz", "cent"], "'return_as' must be 'hz' or 'cent'!"

    if n_freq is None:
        n_freq = int(torch.log2(torch.as_tensor(f_max / f_min)) * 1200 / cent_step)

    k = torch.arange(n_freq)
    f_cent = k * cent_step

    if return_as == "cent":
        return f_cent
    elif return_as == "hz":
        return cents_to_hz(f_cent, f_ref=f_min)


def hz_to_freqidx(f_hz, f_ref=55.0, cent_step=20):
    """Converts frequencies in Hz to indices on a discrete logarithmic frequency axis

    Args:
        f_hz (float or torch.Tensor): Frequency value(s) in Hz
        f_ref (float): Reference frequency in Hz (Default value = 55.0)
        cent_step (float): Resolution of the discrete logarithmic frequency axis in cents

    Returns:
        f_idx (torch.Tensor): Frequency value(s) in cents
    """
    f_cent = hz_to_cents(f_hz, f_ref=f_ref)
    f_idx = torch.round(f_cent / cent_step).type(torch.long)
    return f_idx


def get_f0_targets(
    targets_hz,
    f_min=55.0,
    f_max=1760.0,
    cent_step=10.0,
    n_freq=None,
    gaussian_sigma=None,
    ignore_index=-100,
    return_as="vector",
):
    """Converts frequencies in Hz to optimization targets

    Args:
        targets_hz (float or torch.Tensor): Frequency value(s) in Hz
        f_min (float): Minimum frequency in Hz (Default value = 55.0)
        f_max (float): Maximum frequency in Hz (Default value = 1760.0)
        cent_step (float): Resolution of the discrete logarithmic frequency axis in cents
        n_freq (int): Number of frequency bins (optional, either 'f_max' or 'n_freq' must be None)
        gaussian_sigma (float): Standard deviation used for Gaussian smoothing (optional)
        ignore_index (int): Ignore index to be set for invalid frames
        return_as (str): Specifies whether to return target indices ('index') or target vectors ('vector')

    Returns:
        targets (torch.Tensor): Target tensor
    """
    assert return_as in ["vector", "index"]

    log_frequencies_cent = get_log_frequencies(
        f_min=f_min, f_max=f_max, cent_step=cent_step, n_freq=n_freq, return_as="cent"
    )

    if gaussian_sigma is None:
        targets_idx = hz_to_freqidx(targets_hz, f_ref=f_min, cent_step=cent_step)

        invalid_frames = torch.logical_or(
            targets_idx < 0, targets_idx >= log_frequencies_cent.numel()
        )

        if return_as == "index":
            targets_idx[invalid_frames] = ignore_index
            return targets_idx

        elif return_as == "vector":
            # convert indices into one-hot vectors
            targets_one_hot = torch.zeros(
                *targets_idx.shape,
                log_frequencies_cent.numel(),
                device=targets_idx.device,
            )
            targets_idx[invalid_frames] = 0
            targets_one_hot.scatter_(-1, targets_idx.unsqueeze(dim=-1), 1.0)
            targets_one_hot[invalid_frames] = torch.zeros(
                log_frequencies_cent.numel(), device=targets_idx.device
            )
            return targets_one_hot

    else:
        assert return_as == "vector"

        log_frequencies_cent = log_frequencies_cent.view(*([1] * targets_hz.ndim), -1).to(
            targets_hz.device
        )  # add leading dimensions
        targets_cent = hz_to_cents(targets_hz, f_ref=f_min).unsqueeze(
            dim=-1
        )  # add trailing dimension

        targets_vector = torch.exp(
            -((targets_cent - log_frequencies_cent) ** 2) / (2 * gaussian_sigma**2)
        )
        targets_vector = targets_vector / (
            targets_vector.sum(dim=-1, keepdim=True) + torch.finfo(torch.float32).eps
        )

    return targets_vector
