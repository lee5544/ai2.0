"""ML-specific sample and label selection."""

def generate(*args, **kwargs):
    from .build import generate as build_dataset

    return build_dataset(*args, **kwargs)

__all__ = ["generate"]
