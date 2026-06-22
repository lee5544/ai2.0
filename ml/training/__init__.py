"""ML training orchestration."""

__all__ = ["train_from_config"]


def __getattr__(name: str):
    if name == "train_from_config":
        from ml.train import train_from_config

        return train_from_config
    raise AttributeError(name)
