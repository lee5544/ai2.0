from .add_noise import add_noise
from .random_add_segment import random_add_segment
from .random_amplitude import random_amplitude
from .random_crop import random_crop
from .random_time_stretch import random_time_stretch
from .reverberation import reverberation
from .speed_perturb import speed_perturb
from .time_frequency_mask import time_frequency_mask
from .white_noise import white_noise

METHODS = {
    "add_noise": add_noise,
    "white_noise": white_noise,
    "random_crop": random_crop,
    "random_add_segment": random_add_segment,
    "random_amplitude": random_amplitude,
    "random_time_stretch": random_time_stretch,
    "reverberation": reverberation,
    "time_frequency_mask": time_frequency_mask,
    "speed_perturb": speed_perturb,
}

__all__ = [
    "METHODS",
    "add_noise",
    "random_add_segment",
    "random_amplitude",
    "random_crop",
    "random_time_stretch",
    "reverberation",
    "speed_perturb",
    "time_frequency_mask",
    "white_noise",
]
