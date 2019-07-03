import numpy as np
from numpy import linalg, sqrt, e
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.fftpack import ifft, fft
from typing import Union, List, Tuple, Iterator, Iterable


class WaveletBase:
    '''
    Base class of wavelets.
    You need to write methods to make single wavelet.
    self._make_fft_wavelet : returns np.ndarray
    self._make_wavelet : returns np.ndarray
    '''
    def __init__(self, sfreq: float) -> None:
        self.accuracy = 1
        self.sfreq = sfreq

    def _setup_base_waveshape(self, freq: float,
                              rate: float = 1) -> np.ndarray:
        '''
        Setup wave shape.

        Parameters
        ----------
        freq: float | Base Frequency. For example, 1.
            It must be base frequency.
            You cannot use this for every freqs.

        Returns
        -------
        Tuple[float, float]: (one, total)
        '''

        one: float = 1 / freq / self.accuracy / rate
        total: float = self.sfreq / freq / rate
        return np.arange(0, total, one)

    def _make_fft_wavelet(self, total: float, one: float,
                          freq: float = 1) -> np.ndarray:
        pass

    def _make_fft_wavelets(self, timeline: np.ndarray,
                           freqs: Iterable) -> Iterator:
        '''
        Make Fourier transformed wavelet.
        '''
        return map(lambda freq: self._make_fft_wavelet(timeline, freq),
                   freqs)

    def _make_wavelet(self, freq: float) -> np.ndarray:
        pass

    def make_wavelets(self,
                      freqs: Union[List[float],
                                   range, np.ndarray]) -> List[np.ndarray]:
        '''
        Make wavelets.
        It returnes list of wavelet, and it is compatible with mne-python.
        (As argument of Ws of mne.time_frequency.tfr.cwt)

        Parameters
        ----------
        freqs: List[float] | Frequency. If frequency is too small,
            it returnes bad wave easily.
            For example, sfreq=1000, freq=3 it returnes bad wave.
            If you want good wave, you must set large accuracy, and length
            when you make this instance.

        Returns
        -------
        MorseWavelet: np.ndarray
        '''
        return list(map(self._make_wavelet, freqs))

    def cwt(self, wave: np.ndarray,
            freqs: Union[List[float], range, np.ndarray],
            max_freq: int = 0) -> np.ndarray:
        '''
        Run CWT.
        This method is still experimental.
        It has no error handling code now.
        '''
        wave_length = wave.shape[0]
        rate: float = wave.shape[0] / self.sfreq
        timeline = self._setup_base_waveshape(freqs[0], rate)
        wavelet_base = self._make_fft_wavelets(timeline, freqs)
        wavelet: Iterator = map(lambda w: np.pad(w,
                                                 [0, wave_length - w.shape[0]],
                                                 'constant'),
                                wavelet_base)
        fft_wave = fft(wave)
        result_map: Iterator = map(lambda x: ifft(x * fft_wave),
                                   wavelet)
        max_freq = int(self.sfreq / 2) if max_freq == 0 else max_freq
        result_list = list(result_map)[:max_freq]
        return np.array(result_list)

    def power(self, wave: np.ndarray,
              freqs: Union[List[float], range, np.ndarray]) -> np.ndarray:
        '''
        Run cwt and compute power.

        Parameters
        ----------
        freqs: float | Frequencies. Before use this, please run plot.

        Returns
        -------
        Result of cwt. np.ndarray.
        '''
        result: np.ndarray = self.cwt(wave, freqs)
        return np.abs(result * np.conj(result))

    def plot(self, freq: float, show: bool = True) -> plt.figure:
        wavelet = self.make_wavelets([freq])[0]
        fig = plt.figure(figsize=(6, 8))
        ax = fig.add_subplot(311)
        ax.plot(np.arange(0, wavelet.shape[0], 1),
                wavelet,
                label='morse')
        ax1 = fig.add_subplot(312, projection='3d')
        ax1.scatter3D(wavelet.real,
                      np.arange(0, wavelet.shape[0], 1),
                      wavelet.imag,
                      label='morse')
        ax.set_title('Generalized Morse Wavelet')
        ax2 = fig.add_subplot(313)
        ax2.set_title('Caution')
        ax2.text(0.05, 0.1,
                 'This is inverse Fourier transformed MorseWavelet.\n'
                 'Originally, Morse wavelet is Frourier transformed wave.\n'
                 'It should be used as it is Fourier transformed data.\n'
                 'But, you can use it in the same way as'
                 'MorletWavelet by IFFT.\n'
                 'If wave continues to side of the window, wave is bad.\n'
                 'Please set larger value to param "accuracy" and "length"\n'
                 'It becomes bad easily when frequency is low.')
        ax2.tick_params(labelbottom=False,
                        labelleft=False,
                        labelright=False,
                        labeltop=False,
                        bottom=False,
                        left=False,
                        right=False,
                        top=False)
        if show:
            plt.show()
        fig


