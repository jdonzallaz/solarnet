import torch


def naive_predictions_random_guess(size: int, n_class: int = 2, seed: int = 42) -> torch.Tensor:
    # Predictions with class chosen at random (equal weight) in the classes.
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.randint(n_class, (size,), generator=generator)


def naive_predictions_majority_class(size: int) -> torch.Tensor:
    # Prediction that always choose the first class.
    return torch.full((size,), 0)
