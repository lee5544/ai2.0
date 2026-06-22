from .random_add_segment import random_add_segment
from .random_amplitude import random_amplitude
from .random_crop import random_crop
from .random_time_stretch import random_time_stretch
from .white_noise import white_noise

METHODS = {
    "white_noise": white_noise,
    "random_crop": random_crop,
    "random_add_segment": random_add_segment,
    "random_amplitude": random_amplitude,
    "random_time_stretch": random_time_stretch,
}

__all__ = [
    "METHODS",
    "random_add_segment",
    "random_amplitude",
    "random_crop",
    "random_time_stretch",
    "white_noise",
]
