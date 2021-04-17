import math

import torch

# TODO: parametrize scaling / log-scaling / min/max/mean/std values

# Values computed from the SDO-Benchmark dataset on peak_flux values.
MEAN = 2.8510188178831223e-06
STD = 1.4931222734709925e-05
MIN = 1e-9
MAX = 0.0011


# ===== Base scaling =====

def min_max_scale(t, tmin, tmax):
    # Min-max scaling in [0,1] range from [tmin,tmax] range.
    return (t - tmin) / (tmax - tmin)


def min_max_inverse_scale(t, tmin, tmax):
    # Min-max inverse scaling from [0,1] range to [tmin,tmax] range.
    return t * (tmax - tmin) + tmin


def standard_scale(t, mean, std):
    # Standard scaling using mean/std.
    return (t - mean) / std


def standard_inverse_scale(t, mean, std):
    # Standard inverse scaling using mean/std.
    return t * std + mean


def log_scale_value(v: float) -> float:
    # Scale a value from log scale to normal scale
    return math.log10(v)


def log_inverse_scale_tensor(t: torch.Tensor) -> torch.Tensor:
    # Scale a tensor from normal scale to log scale
    return torch.pow(torch.full_like(t, 10), t)


# ===== Utility functions for scaling =====

def simple_standard_scale(t):
    # Standard scale a value/tensor
    return standard_scale(t, MEAN, STD)


def simple_min_max_scale(t):
    # Min-max scale a value/tensor
    return min_max_scale(t, MIN, MAX)


def simple_standard_inverse_scale(t: torch.Tensor) -> torch.Tensor:
    # Standard inverse scale a tensor
    return standard_inverse_scale(t, MEAN, STD)


def simple_min_max_inverse_scale(t: torch.Tensor) -> torch.Tensor:
    # Min-max inverse scale a value/tensor
    return min_max_inverse_scale(t, MIN, MAX)


# ===== Utility functions for scaling and log-scaling =====


def log_standard_scale(t):
    # Log and standard scale a value
    return standard_scale(log_scale_value(t), log_scale_value(MEAN), log_scale_value(STD))


def log_min_max_scale(t):
    # Log and min-max scale a value
    return min_max_scale(log_scale_value(t), log_scale_value(MIN), log_scale_value(MAX))


def log_standard_inverse_scale(t: torch.Tensor) -> torch.Tensor:
    # Log and standard inverse scale a tensor
    return log_inverse_scale_tensor(standard_inverse_scale(t, log_scale_value(MEAN), log_scale_value(STD)))


def log_min_max_inverse_scale(t: torch.Tensor) -> torch.Tensor:
    # Log and min-max inverse scale a value/tensor
    return log_inverse_scale_tensor(min_max_inverse_scale(t, log_scale_value(MIN), log_scale_value(MAX)))
