import numpy as np
import cupy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.fftpack import ifft, fft
from typing import Union, List, Tuple, Iterator, Iterable
from enum import Enum


def kill_nyquist(wave: np.ndarray) -> np.ndarray:
    half_size = int(wave.shape[0] / 2)
    wave = np.pad(wave[:half_size],
                  [0, wave.shape[0] - half_size],
                  'constant')
    return wave


def nin_fft(wave: np.ndarray) -> np.ndarray:
    return kill_nyquist(fft(wave))


class WaveletMode(Enum):
    Normal = 0
    Both = 1
    Reverse = 2
    Indifferentiable = 3


class WaveletBase:
    '''
    Base class of wavelets.
    You need to write methods to make single wavelet.
    self._make_fft_wavelet : returns np.ndarray
    self.make_wavelet : returns np.ndarray
    '''
    def __init__(self, sfreq: float) -> None:
        self.mode = WaveletMode.Normal
        self.accuracy: float = 1
        self.sfreq = sfreq
        self.length = 10
        self.help = ''
        self.use_cuda = False

    def _setup_base_trans_waveshape(self, freq: float,
                                    real_length: float = 1) -> np.ndarray:
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
        one: float = 1 / freq / self.accuracy / real_length
        total: float = self.sfreq / freq / real_length
        return np.arange(0, total, one)

    def _setup_base_waveletshape(self, freq: float,
                                 real_length: float = 1) -> np.ndarray:
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
        total: float = real_length / self.peak_freq(freq) * freq * 2 * np.pi
        one = 1 / self.sfreq * 2 * np.pi * freq / self.peak_freq(freq)
        return np.arange(0, total, one)

    def peak_freq(self, freq: float) -> float:
        return 0.

    def _normalize(self, wave: np.ndarray) -> np.ndarray:
        wave /= np.linalg.norm(wave.ravel()) * np.sqrt(0.5)
        return wave

    def make_fft_wavelet(self, base_freq: float, real_wave_length: float,
                         freq: float = 1) -> np.ndarray:
        if self.mode in [WaveletMode.Reverse, WaveletMode.Both]:
            timeline = self._setup_base_trans_waveshape(base_freq,
                                                        real_wave_length)
            trans_wavelet = self.trans_wavelet_formula(timeline, freq)
            return self._normalize(trans_wavelet)
        else:
            wavelet = self.make_wavelet(freq)
            wavelet = np.pad(wavelet,
                             int((self.sfreq * real_wave_length
                                  - wavelet.shape[0]) / 2),
                             'constant')
            result = self._normalize(nin_fft(wavelet) / self.sfreq)
            result = kill_nyquist(result)
            return result

    def make_fft_wavelets(self, base_freq: float, real_wave_length: float,
                          freqs: Iterable) -> Iterator:
        '''
        Make Fourier transformed wavelet.
        '''
        return map(lambda freq: self.make_fft_wavelet(base_freq,
                                                      real_wave_length, freq),
                   freqs)

    def wavelet_formula(self, timeline: np.ndarray, freq: float) -> np.ndarray:
        return timeline

    def trans_wavelet_formula(self, timeline: np.ndarray,
                              freq: float = 1.) -> np.ndarray:
        return timeline

    def make_wavelet(self, freq: float) -> np.ndarray:
        if self.mode == WaveletMode.Reverse:
            timeline = self._setup_base_trans_waveshape(freq)
            wave = self.trans_wavelet_formula(timeline)
            wavelet: np.ndarray = ifft(wave)
        else:
            timeline = self._setup_base_waveletshape(freq)
            wavelet = self.wavelet_formula(timeline, freq)
        half = int(wavelet.shape[0])
        band = int(half / 2 / freq * self.length)
        start: int = half - band if band < half // 2 else half // 2
        stop: int = half + band if band < half // 2 else half // 2 * 3
        # cut side of wavelets and contactnate
        total_wavelet = np.hstack((np.conj(np.flip(wavelet)),
                                   wavelet))
        wavelet = total_wavelet[start: stop]
        # normalize
        self._normalize(wavelet)
        return wavelet

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
        return list(map(self.make_wavelet, freqs))

    def cwt(self, wave: np.ndarray,
            freqs: Union[List[float], range, np.ndarray],
            max_freq: int = 0) -> np.ndarray:
        '''cwt
        Run CWT.
        This method is still experimental.

        wave:
        freqs:
        max_freq:
        '''
        freq_dist = freqs[1] - freqs[0]
        wave_length = wave.shape[0]
        real_wave_length: float = wave.shape[0] / self.sfreq
        base_freq = 1
        # base_freq = wave.shape[0]
        wavelet_base = self.make_fft_wavelets(base_freq,
                                              real_wave_length,
                                              freqs)
        wavelet: Iterator = map(lambda w: np.pad(w,
                                                 [0, wave_length - w.shape[0]],
                                                 'constant'),
                                wavelet_base)
        if self.use_cuda:
            fft_wave = cupy.fft.fft(cupy.asarray(wave))
            result_map: Iterator = map(lambda x: cupy.asnumpy(cupy.fft.ifft(cupy.asarray(x) * fft_wave)),
                                       wavelet)
            if max_freq == 0:
                max_freq = int(self.sfreq / freq_dist)
            result_list = list(result_map)[:max_freq]
            return np.array(result_list)
        else:
            fft_wave = nin_fft(wave)
            result_map = map(lambda x: ifft(x * fft_wave),
                             wavelet)
            if max_freq == 0:
                max_freq = int(self.sfreq / freq_dist)
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
        return np.abs(result)

    def plot(self, freq: float, show: bool = True) -> plt.figure:
        if self.help == '':
            plt_num = 3
        wavelet = self.make_wavelets([freq])[0]
        fig = plt.figure(figsize=(6, 8))
        ax = fig.add_subplot(plt_num, 1, 1)
        ax.plot(np.arange(0, wavelet.shape[0], 1),
                wavelet,
                label='morse')
        ax1 = fig.add_subplot(plt_num, 1, 2, projection='3d')
        ax1.scatter3D(wavelet.real,
                      np.arange(0, wavelet.shape[0], 1),
                      wavelet.imag,
                      label='morse')
        ax.set_title('Generalized Morse Wavelet')
        if plt_num == 3:
            ax2 = fig.add_subplot(313)
            ax2.set_title('Caution')
            ax2.text(0.05, 0.1, self.help)
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


