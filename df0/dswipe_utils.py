from functools import lru_cache
import torch


@lru_cache(maxsize=16)
def interp1d_linear_get_matrix(x_old, x_new):
    """
    Computes a matrix to perform linear interpolation via matrix multiplication.

    Parameters:
        x_old (list): x-coordinates of the given data points (monotonically increasing)
        x_new (list): x-coordinates at which to evaluate the interpolated values

    Returns:
        interpolation_matrix (torch.Tensor): interpolation matrix
    """
    x_old = torch.as_tensor(x_old)
    x_new = torch.as_tensor(x_new)
    interpolation_matrix = torch.zeros(x_old.numel(), x_new.numel())

    for i, x_curr in enumerate(x_new):
        # find index of first value in x_old that is larger than x_curr
        idx_ceil = int(torch.argmax((x_old > x_curr.cpu()).float()).item())

        # check whether x_curr identical to largest x_old
        if x_curr == x_old[-1]:
            idx_ceil = x_old.numel() - 1

        idx_floor = idx_ceil - 1

        # determine relative position on the interval between idx_floor and idx_ceil
        alpha = (x_curr - x_old[idx_floor]) / (x_old[idx_ceil] - x_old[idx_floor])

        if idx_ceil < x_old.numel() and idx_floor >= 0:
            interpolation_matrix[idx_floor, i] = 1 - alpha
            interpolation_matrix[idx_ceil, i] = alpha

    return interpolation_matrix


def interp1d_linear(y, x_old=None, x_new=None, dim=0):
    """
    Performs linear interpolation of the values in a tensor y.

    Parameters:
        y (torch.Tensor): tensor to be interpolated
        x_old (list): x-coordinates of the given data points (monotonically increasing)
        x_new (list): x-coordinates at which to evaluate the interpolated values
        dim (int): dimension across which to interpolate

    Returns:
        y_interp (torch.Tensor): interpolated tensor
    """
    assert y.shape[dim] == len(x_old), "Dimensionality does not match. Check inputs!"

    interpolation_matrix = interp1d_linear_get_matrix(x_old, x_new).to(y.device)

    y = torch.transpose(y, dim, -1)
    y_interp = torch.matmul(y, interpolation_matrix)
    return torch.transpose(y_interp, dim, -1)


def prime_and_one(upto=1000000):
    """
    This function returns a tensor containing 1 and all the prime number up to the given argument.

    Adapted from: https://github.com/groupmm/libf0/blob/main/libf0/swipe_slim.py

    Parameters:
        upto (int): upper limit for the prime numbers to return

    Returns:
        primes_and_one (torch.Tensor): tensor containining 1 and prime numbers
    """
    primes = torch.arange(3, upto + 1, 2)
    isprime = torch.ones((upto - 1) // 2, dtype=torch.bool)
    for factor in primes[: int(torch.sqrt(torch.tensor([upto]))) // 2]:
        if isprime[(factor - 2) // 2]:
            isprime[(factor * 3 - 2) // 2 :: factor] = 0
    return torch.cat((torch.tensor([1, 2]), primes[isprime]))
