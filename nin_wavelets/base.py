import numpy as np
import cupy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.fftpack import ifft, fft
from typing import Union, List, Tuple, Iterator, Iterable
from enum import Enum
from mpl_toolkits.axes_grid1 import make_axes_locatable


Numbers = Union[List[float], np.ndarray, range]


def interpolate_alias(wave: np.ndarray) -> np.ndarray:
    '''
    Kill wave over Nyquist frequency.
    Not a method to kill Mr Nyquist, I am sorry.
    '''
    half_size: int = int(wave.shape[0] / 2)
    wave = np.pad(wave[:half_size],
                  [0, wave.shape[0] - half_size],
                  'constant', constant_values=0)
    return wave * 2


class WaveletMode(Enum):
    '''
    Modes of Wavelets.
    These are used as Wavelet.mode
    '''
    Normal = 0
    # Use Wavelet fromula only
    Both = 1
    # Use Wavwlet formula and FFTed formula.
    Reverse = 2
    # Use FFTed formula only
    Indifferentiable = 3
    # Indifferentiable formula
    Twice = 4
    # Even if FFTed formula is there,
    # Use IFFTed Wavelet, and FFT.
    # This is ugly and not accurate.


class WaveletBase:
    '''
    Base class of wavelets.
    You need to write methods to make single wavelet.
    self._make_fft_wavelet : returns np.ndarray
    self.make_wavelet : returns np.ndarray
    '''
    def __init__(self, sfreq: float = 1000, accuracy: float = 1.,
                 real_wave_length: float = 1.,
                 interpolate: bool = False) -> None:
        '''
        Parameters
        ----------
        sfreq: float|
            Sampling frequency.
        accuracy: float|
            This value affects only when you plot the wavelet or
            you make wavelet in depricated way.
        real_wave_length: float|
            Length of wavelet. When this class run cwt,
            this will be automatically changed.
        '''
        self.mode = WaveletMode.Normal
        self.accuracy = accuracy
        self.sfreq = sfreq
        self.help = ''
        self.real_wave_length = real_wave_length
        self.interpolate = interpolate

    def _setup_base_trans_waveshape(self, freq: float,
                                    real_length: float = 1) -> np.ndarray:
        '''
        Setup wave shape.
        real_length is length of wavelet(for example, sec or msec)
        self.real_wave_length is length of wave to analyze.

        Parameters
        ----------
        freq: float | Base Frequency. For example, 1.
            It must be base frequency.
            You cannot use this for every freqs.

        Returns
        -------
        np.ndarray | Timeline to calculate wavelet.
        '''
        one: float = 1 / freq / self.accuracy / real_length
        total: float = self.sfreq / freq / real_length * self.real_wave_length
        return np.arange(0, total, one, dtype=np.float)

    def _setup_base_waveletshape(self, freq: float, real_length: float = 1,
                                 zero_mean: bool = False) -> np.ndarray:
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
        one: float = 1 / self.sfreq * 2 * np.pi * freq / self.peak_freq(freq)
        if zero_mean:
            return np.arange(-total / 2, total / 2, one)
        return np.arange(0, total, one)

    def peak_freq(self, freq: float) -> float:
        return 1.

    def _normalize(self, wave: np.ndarray) -> np.ndarray:
        ''' Normalize norm of complex array

        Parameters
        ----------
        wave: np.ndarray[np.complex128, ndim=1] |
            Wave to normalize.

        Returns
        -------
        np.ndarray[np.complex128, ndim=1]: Normalized wave.
        '''
        wave /= np.linalg.norm(wave.ravel()) * np.sqrt(0.5)
        return wave

    def make_fft_wavelet(self, freq: float = 1.) -> np.ndarray:
        ''' Make single FFTed wavelet.

        Parameters
        ----------
        freq: float | Frequency of wavelet.

        Returns
        -------
        np.ndarray[np.complex128, ndim=1]: FFTed Wavelet.
        '''
        if self.mode in [WaveletMode.Reverse, WaveletMode.Both]:
            timeline = self._setup_base_trans_waveshape(self.real_wave_length)
            result = np.asarray(self.trans_wavelet_formula(timeline, freq),
                                dtype=np.complex128)
            result = interpolate_alias(result) if self.interpolate else result
            return self._normalize(result)
        else:
            wavelet = self.make_wavelet(freq)
            wavelet = wavelet.astype(np.complex128)
            half = int((self.sfreq *
                        self.real_wave_length - wavelet.shape[0]) / 2)

            wavelet = np.hstack((np.zeros(half, dtype=np.complex128),
                                 wavelet,
                                 np.zeros(half, dtype=np.complex128)))
            wavelet = wavelet.astype(np.complex128)
            result = fft(wavelet) / self.sfreq
            result.imag = np.abs(result.imag)
            result.real = np.abs(result.real)
            result = interpolate_alias(result) if self.interpolate else result
            result = self._normalize(result)
            return result

    def make_fft_wavelets(self, freqs: Numbers) -> List[np.ndarray]:
        ''' Make list of FFTed wavelets.
        Make Fourier transformed wavelet.

        Parameters
        ----------
        freq: float | Frequency of wavelet.

        Returns
        -------
        np.ndarray[np.complex128, ndim=1]: FFTed Wavelet.
        '''
        self.fft_wavelets = list(map(self.make_fft_wavelet, freqs))
        return self.fft_wavelets

    def wavelet_formula(self, timeline: np.ndarray, freq: float) -> np.ndarray:
        ''' wavelet_formula
        The formula of Wavelet.
        Other procedures are performed by other methods.

        Parameters
        ----------
        timeline: np.ndarray[np.float, ndim=1]|
            Time value of formula.
        freq: float|
            If you want to setup peak frequency,
            this variable may be useful.

        Returns
        -------
        Base of wavelet.
            timeline: np.ndarray:

        freq: float:
        '''
        return timeline

    def trans_wavelet_formula(self, freqs: Iterator[float],
                              freq: float = 1.) -> np.ndarray:
        ''' trans_wavelet_formula
        The formula of Fourier Transformed Wavelet.
        Other procedures are performed by other methods.

        Parameters
        ----------
        freqs: np.ndarray[np.float, ndim=1]|
            Frequencies.
            If length of time is same as freqs, It is easy to write.
        freq: float|
            If you want to setup peak frequency,
            this variable may be useful.

        Returns
        -------
        Base of wavelet.
            freqs: np.ndarray:

        freq: float:
        '''
        return freqs

    def make_wavelet(self, freq: float) -> np.ndarray:
        if self.mode in [WaveletMode.Reverse, WaveletMode.Twice]:
            timeline: np.ndarray = self._setup_base_trans_waveshape(freq)
            wave = self.trans_wavelet_formula(timeline)
            wavelet: np.ndarray = ifft(wave)
            half: int = int(wavelet.shape[0])
            # start: int = half - band if band < half // 2 else half // 2
            # stop: int = half + band if band < half // 2 else half // 2 * 3
            start: int = half // 2
            stop: int = half // 2 * 3
            # cut side of wavelets and contactnate
            total_wavelet = np.hstack((np.conj(np.flip(wavelet)),
                                       wavelet))
            wavelet = total_wavelet[start: stop]
        else:
            timeline = self._setup_base_waveletshape(freq, 1, zero_mean=True)
            wavelet = np.asarray(self.wavelet_formula(timeline, freq),
                                 dtype=np.complex128)
        return self._normalize(wavelet)

    def make_wavelets(self,  freqs: Numbers) -> np.ndarray:
        '''
        Make wavelets.
        It returnes list of wavelet, and it is compatible with mne-python.

        Parameters
        ----------
        freqs: List[float] | Frequencies.

        Returns
        -------
        MorseWavelet: np.ndarray
        '''
        self.wavelets = list(map(self.make_wavelet, freqs))
        return self.wavelets

    def cwt(self, wave: np.ndarray,
            freqs: Numbers,
            max_freq: int = 0) -> np.ndarray:
        '''cwt
        Run CWT.

        wave: np.ndarray| Wave to analyze
        freqs: Union[List[float], range, np.ndarray]|
            Frequencies
        max_freq: int| Max Frequency
        '''
        # =====================================
        # This section should be cut
        # =====================================
        freq_dist: float = freqs[1] - freqs[0]
        wave_length: int = wave.shape[0]
        self.real_wave_length: float = wave.shape[0] / self.sfreq
        wavelet_base = self.make_fft_wavelets(freqs)
        wavelet = map(lambda x: np.pad(x, [0, wave_length - x.shape[0]],
                                       'constant'),
                      wavelet_base)
        # =====================================
        fft_wave = fft(wave)
        if self.interpolate:
            fft_wave = interpolate_alias(fft_wave) / 2
        else:
            fft_wave
        # Keep powerful even if long wave.
        fft_wave *= (wave_length / self.sfreq) ** 0.5
        result_map = map(lambda x: ifft(x * fft_wave), wavelet)
        if max_freq == 0:
            max_freq = int(self.sfreq / freq_dist)
        result_list = list(result_map)[:max_freq]
        # reset myself
        self.real_wave_length = 1.
        return np.array(result_list)

    def power(self, wave: np.ndarray, freqs: Numbers) -> np.ndarray:
        '''
        Run cwt and compute power.

        Parameters
        ----------
        wave: np.ndarray| Wave to analyze
        freqs: float | Frequencies. Before use this, please run plot.

        Returns
        -------
        Result of cwt. np.ndarray.
        '''
        result = self.cwt(wave, freqs)
        return np.abs(result) ** 2

    def abs(self, wave: np.ndarray,
            freqs: Numbers) -> np.ndarray:
        '''
        Run cwt and compute power.

        Parameters
        ----------
        wave: np.ndarray| Wave to analyze
        freqs: float | Frequencies. Before use this, please run plot.

        Returns
        -------
        Result of cwt. np.ndarray.
        '''
        result = self.cwt(wave, freqs)
        return np.abs(result)

    def plot(self, freq: float, show: bool = True) -> plt.figure:
        '''
        Plot wavelet.

        Parameters
        ----------
        freq: float | Frequency of Wavelet.
        show: bool| Show plot.

        Returns
        -------
        Fig of matplotlib.
        '''
        freqs = np.array([freq])
        plt_num = 3 if self.help else 2
        wavelet = self.make_wavelets(freqs)[0]
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
        return fig


def plot_tf(data, vmin: float = None, vmax: float = None,
            cmap: str = 'RdBu_r', show: bool = True) -> plt.Axes:
    '''
    Plot by matplotlib.
    vrange: Tuple[float, float]|
        This is range of color.
        Same as tuple of vmin and vmax of matplotlib.
    '''
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_aspect('auto')
    image = ax.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.invert_yaxis()
    divider = make_axes_locatable(ax)
    ax_cb = divider.new_horizontal(size="2%", pad=0.05)
    fig.add_axes(ax_cb)
    plt.colorbar(image, cax=ax_cb)
    if show:
        plt.show()
    return ax

