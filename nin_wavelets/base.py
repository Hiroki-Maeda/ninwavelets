import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import ifft, fft
from typing import Union, List, Iterator, Type
from enum import Enum
from os import cpu_count
from mpl_toolkits.axes_grid1 import make_axes_locatable


Numbers = Union[List[float], np.ndarray, range]


def hamming_window(wave: np.ndarray) -> np.ndarray:
    length = wave.shape[0]
    window = np.arange(0, 1, 1 / length)
    return 0.54 - 0.46 * np.cos(2 * np.pi * window)


def normalize(wave: np.ndarray) -> np.ndarray:
    ''' Normalize norm of complex array

    Parameters
    ----------
    wave: np.ndarray[np.complex128, ndim=1]
        Wave to normalize.

    Returns
    -------
    np.ndarray[np.complex128, ndim=1]: Normalized wave.
    '''
    wave /= np.linalg.norm(wave.ravel()) * np.sqrt(0.5)
    return wave


def interpolate_alias(wave: np.ndarray) -> np.ndarray:
    '''
    Interpolate data over nyquist frequency.

    Parameters
    ----------
    wave: np.ndarray[np.complex128, ndim=1]
        Wave to interpolate.

    Returns
    -------
    np.ndarray[np.complex128, ndim=1]: Interpolated wave.
    '''
    half_size: int = int(wave.shape[0] / 2)
    wave = np.pad(wave[:half_size],
                  [0, wave.shape[0] - half_size],
                  'constant', constant_values=0)
    return wave


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
                 interpolate: bool = True) -> None:
        '''
        Parameters
        ----------
        sfreq: float
            Sampling frequency.
        accuracy: float
            This value affects only when you plot the wavelet or
            you make wavelet in depricated way.
        real_wave_length: float
            Length of wavelet. When this class run cwt,
            this will be automatically changed.
        '''
        self.mode: WaveletMode = WaveletMode.Normal
        self.accuracy: float = accuracy
        self.sfreq: float = sfreq
        self.help: str = ''
        self.real_wave_length: float = real_wave_length
        # Distance between freqs(cwt)
        self.freq_dist: float
        self.interpolate = interpolate

    def _setup_base_trans_waveshape(self, freq: float,
                                    real_wave_length: float) -> np.ndarray:
        '''
        Setup wave shape.
        real_length is length of wavelet(for example, sec or msec)
        self.real_wave_length is length of wave to analyze.

        Parameters
        ----------
        freq: float
            Base Frequency. For example, 1.
            It must be base frequency.
            You cannot use this for every freqs.

        Returns
        -------
        np.ndarray
            Timeline to calculate wavelet.
        '''
        one: float = 1 / freq / self.accuracy
        total: float = self.sfreq / freq * real_wave_length
        return np.arange(0, total, one, dtype=np.float)

    def _setup_base_waveletshape(self, freq: float, real_length: float = 1,
                                 zero_mean: bool = False) -> np.ndarray:
        '''
        Setup wave shape.

        Parameters
        ----------
        freq: float
            Base Frequency. For example, 1.
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

    def make_fft_wavelet(self, freq: float = 1.) -> np.ndarray:
        ''' Make single FFTed wavelet.

        Parameters
        ----------
        freq: float
            Frequency of wavelet.

        Returns
        -------
        np.ndarray[np.complex128, ndim=1]: FFTed Wavelet.
        '''
        if self.mode in [WaveletMode.Reverse, WaveletMode.Both]:
            if self.interpolate:
                t = self._setup_base_trans_waveshape(self.real_wave_length,
                                                     self.real_wave_length / 2)
                result = self.trans_wavelet_formula(t, freq)
                result = np.hstack((result, np.zeros(len(t))))
            else:
                t = self._setup_base_trans_waveshape(self.real_wave_length,
                                                     self.real_wave_length)
                result = self.trans_wavelet_formula(t, freq)
            return normalize(result)
        else:
            wavelet = self.make_wavelet(freq)
            half = int((self.sfreq * self.real_wave_length
                        - wavelet.shape[0]) / 2)
            wavelet = np.hstack((np.zeros(half), wavelet, np.zeros(half)))
            result = fft(wavelet) / self.sfreq
            result.imag = np.abs(result.imag)
            result.real = np.abs(result.real)
            return normalize(result)

    def make_fft_wavelets(self, freqs: Numbers) -> List[np.ndarray]:
        ''' Make list of FFTed wavelets.
        Make Fourier transformed wavelet.

        Parameters
        ----------
        freq: float
            Frequency of wavelet.

        Returns
        -------
        np.ndarray[np.complex128, ndim=1]: FFTed Wavelet.
        '''
        self.freq_dist = freqs[1] - freqs[0]
        if self.interpolate:
            self.fft_wavelets = list(map(interpolate_alias,
                                         map(self.make_fft_wavelet,
                                             freqs))
                                     )
        else:
            self.fft_wavelets = list(map(self.make_fft_wavelet,
                                         freqs))
        return self.fft_wavelets

    def wavelet_formula(self, timeline: np.ndarray, freq: float) -> np.ndarray:
        ''' wavelet_formula
        The formula of Wavelet.
        Other procedures are performed by other methods.

        Parameters
        ----------
        timeline: np.ndarray[np.float, ndim=1]
            Time value of formula.
        freq: float
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
        freqs: np.ndarray[np.float, ndim=1]
            Frequencies.
            If length of time is same as freqs, It is easy to write.
        freq: float
            If you want to setup peak frequency,
            this variable may be useful.

        Returns
        -------
        Base of wavelet: np.ndarray:
        '''
        return freqs

    def make_wavelet(self, freq: float) -> np.ndarray:
        if self.mode in [WaveletMode.Reverse, WaveletMode.Twice]:
            t = self._setup_base_trans_waveshape(freq, self.real_wave_length)
            wave = self.trans_wavelet_formula(t)
            wavelet = ifft(wave)
            half = int(wavelet.shape[0])
            start = half // 2
            stop = half // 2 * 3
            total_wavelet = np.hstack((np.conj(np.flip(wavelet)), wavelet))
            wavelet = total_wavelet[start: stop]
        else:
            timeline = self._setup_base_waveletshape(freq, 1, zero_mean=True)
            wavelet = self.wavelet_formula(timeline, freq)
        return normalize(wavelet)

    def make_wavelets(self,  freqs: Numbers) -> np.ndarray:
        '''
        Make wavelets.
        It returnes list of wavelet, and it is compatible with mne-python.

        Parameters
        ----------
        freqs: List[float]
            Frequencies.

        Returns
        -------
        MorseWavelet: np.ndarray
        '''
        self.wavelets = list(map(self.make_wavelet, freqs))
        return self.wavelets

    def cwt(self, wave: np.ndarray,
            freqs: Union[Numbers, None], max_freq: int = 0,
            reuse: bool = True) -> np.ndarray:
        '''cwt
        Run CWT.

        wave: np.ndarray
            Wave to analyze
        freqs: Union[List[float], range, np.ndarray]
            Frequencies. It can be argument of cwt, but it is slow.
            If you want to calculate repeatedly, you should run
            make_fft_wavelets before cwt, and freqs should be None.
        max_freq: int
            Max Frequency
        reuse: bool
            Use wavelet which was made before.
        '''
        self.real_wave_length: float = wave.shape[0] / self.sfreq
        if (not reuse) or (not hasattr(self, 'fft_wavelets')):
            self.make_fft_wavelets(freqs)
        wavelet = [np.pad(x, [0, wave.shape[0] - x.shape[0]], 'constant')
                   for x in self.fft_wavelets]
        fft_wave = fft(wave)
        if self.interpolate:
            fft_wave = interpolate_alias(fft_wave)
        # Keep powerful even if long wave.
        fft_wave *= np.sqrt(wave.shape[0] / self.sfreq)
        # result = [ifft(x * fft_wave) for x in wavelet]
        result = ifft(wavelet * fft_wave)
        if max_freq == 0:
            max_freq = int(self.sfreq / self.freq_dist)
        self.real_wave_length = 1.
        return result[:max_freq]

    def power(self, wave: np.ndarray,
              freqs: Union[Numbers, None] = None, max_freq: int = 0,
              reuse: bool = True) -> np.ndarray:
        '''
        Run cwt and compute power.

        Parameters
        ----------
        wave: np.ndarray
            Wave to analyze
        freqs: float
            Frequencies. Before use this, please run plot.

        Returns
        -------
        Result of cwt. np.ndarray.
        '''
        return self.abs(wave, freqs, max_freq, reuse) ** 2

    def abs(self, wave: np.ndarray,
            freqs: Union[Numbers, None] = None, max_freq: int = 0,
            reuse: bool = True) -> np.ndarray:
        '''
        Run cwt and compute power.

        Parameters
        ----------
        wave: np.ndarray
            Wave to analyze
        freqs: float
            Frequencies. Before use this, please run plot.

        Returns
        -------
        Result of cwt. np.ndarray.
        '''
        return np.abs(self.cwt(wave, freqs, max_freq, reuse))

    def plot(self, freq: float, show: bool = True) -> plt.figure:
        return plot_wavelet(self, freq, show)


def plot_wavelet(wavelet_obj: Type[WaveletBase], freq: float,
                 show: bool = True) -> plt.figure:
    '''
    Plot wavelet.

    Parameters
    ----------
    freq: float
        Frequency of Wavelet.
    show: bool
        Show plot.

    Returns
    -------
    Fig of matplotlib.
    '''
    freqs = np.array([freq])
    plt_num = 3 if wavelet_obj.help else 2
    wavelet = wavelet_obj.make_wavelets(freqs)[0]
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
        ax2.text(0.05, 0.1, wavelet_obj.help)
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


def plot_tf(data: np.ndarray, vmin: Union[float, None] = None,
            vmax: Union[float, None] = None,
            cmap: str = 'RdBu_r', show: bool = True) -> plt.Axes:
    '''
    Plot by matplotlib.
    vrange: Tuple[float, float]
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
