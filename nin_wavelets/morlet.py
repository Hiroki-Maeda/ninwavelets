from .wavelets import WaveletBase
from typing import Tuple, Union, List, Iterable, Iterator
import numpy as np
from mne.time_frequency import tfr


class Morlet(WaveletBase):
    '''
    Base class of wavelets.
    '''
    def __init__(self, sfreq: float, sigma: float = 7.) -> None:
        self.sfreq = sfreq
        self.sigma = sigma
        self.c = np.sqrt(1 +
                         np.e ** (-self.sigma ** 2 / 2) -
                         2 * np.e ** (-3 / 4 * self.sigma ** 2))

    def _make_fft_wavelet(self, total: float, one: float,
                          freq: float = 1) -> np.ndarray:
        pass

