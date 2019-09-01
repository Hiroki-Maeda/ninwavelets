import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from scipy.fftpack import ifft, fft
from typing import Union, List, Iterator, Callable
from enum import Enum
from mpl_toolkits.axes_grid1 import make_axes_locatable
from functools import partial


Numbers = Union[List[float], np.ndarray, range]
Array = Union[np.ndarray, cp.ndarray]
MNE_CONSTANT = np.sqrt(2)


def baseline_of(wave: Array, sfreq: float,
                start: float, stop: float) -> Array:
    return wave[int(start * sfreq): int(stop * sfreq)]


class Baseline:
    def __init__(self, wave: Array, sfreq: float,
                 start: float, stop: float) -> None:
        self.wave = wave
        self.baseline = wave[int(start * sfreq): int(stop * sfreq)]
        self.basemean = self.baseline.mean()

    def mean(self) -> Array:
        return self.wave - self.basemean

    def ratio(self) -> Array:
        return self.wave / self.basemean

    def percent(self) -> Array:
        return self.mean() / self.basemean

    def log(self) -> Array:
        return np.log10(self.ratio())

    def zscore(self) -> Array:
        return self.mean() / np.std(self.baseline)

    def zlog(self) -> Array:
        return self.log() / np.std(self.baseline)


class SizeError(BaseException):
    def __init__(self, err: str) -> None: print(err)


def pad_to(wave_from: np.ndarray, wave_to: np.ndarray) -> np.ndarray:
    from_size, to_size = wave_from.shape[0], wave_to.shape[0]
    if from_size > to_size:
        return wave_from[:to_size]
    else:
        side1 = (to_size - from_size) // 2
        side2 = to_size - from_size - side1
        return np.pad(wave_from, [side1, side2], 'constant')


def hamming_window(wave: np.ndarray) -> np.ndarray:
    length = wave.shape[0]
    window = np.arange(0, 1, 1 / length)
    return 0.54 - 0.46 * np.cos(2 * np.pi * window)


def normalize(wave: np.ndarray, length: float,
              cuda: bool = False) -> np.ndarray:
    ''' Normalize norm of complex array

    Parameters
    ----------
    wave: np.ndarray[np.complex128, ndim=1]
        Wave to normalize.

    Returns
    -------
    np.ndarray[np.complex128, ndim=1]: Normalized wave.
    '''
    return wave * length / np.linalg.norm(wave)


def interpolate_alias(wave: Union[cp.ndarray, np.ndarray],
                      cuda: bool = False) -> np.ndarray:
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
    half: int = int(wave.shape[0] / 2)
    pad: Callable = cp.pad if cuda else np.pad
    return pad(wave[:half], [0, wave.shape[0] - half], 'constant')


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

    def __init__(self, sfreq: float = 1000, real_wave_length: float = 1.,
                 interpolate: bool = True, cuda: bool = False) -> None:
        '''
        Parameters
        ----------
        sfreq: float
            Sampling frequency.
        real_wave_length: float
            Length of wavelet. When this class run cwt,
            this will be automatically changed.
        '''
        self.mode: WaveletMode = WaveletMode.Normal
        self.sfreq: float = sfreq
        self.help: str = ''
        self.real_wave_length: float = real_wave_length
        # Distance between freqs(cwt)
        self.freq_dist: float
        self.interpolate = interpolate
        self.cuda = cuda

    def _setup_trans_shape(self, freq: float, real_wave_length: float,
                           cuda: float) -> np.ndarray:
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
        one: float = 1 / freq
        total: float = self.sfreq / freq * real_wave_length
        return cp.arange(0, total, one) if cuda else np.arange(0, total, one)

    def _setup_waveletshape(self, freq: float, real_length: float = 1,
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

    def make_fft_wavelet(self, freq: float,
                         real_length: float = 1.) -> np.ndarray:
        ''' Make single FFTed wavelet.

        Parameters
        ----------
        freq: float
            Frequency of wavelet.

        Returns
        -------
        np.ndarray[np.complex128, ndim=1]: FFTed Wavelet.
        '''
        if freq == 0:
            raise ZeroDivisionError
        hstack = cp.hstack if self.cuda else np.hstack
        formula = self.cp_trans_formula if self.cuda else self.trans_formula
        if self.mode in [WaveletMode.Reverse, WaveletMode.Both]:
            if self.interpolate:
                t = self._setup_trans_shape(real_length,
                                            real_length / 2, self.cuda)
                result = hstack((formula(t, freq), np.zeros(len(t))))
            else:
                t = self._setup_trans_shape(real_length,
                                            real_length, self.cuda)
                result = formula(t, freq)
            result = cp.asnumpy(result) if self.cuda else result
            return normalize(result, self.sfreq/1000)
        else:
            wavelet = self.make_wavelet(freq)
            half = int((self.sfreq * self.real_wave_length
                        - wavelet.shape[0]) / 2)
            wavelet = np.hstack((np.zeros(half), wavelet, np.zeros(half)))
            result = fft(wavelet)
            result.imag, result.real = np.abs(result.imag), np.abs(result.real)
            return normalize(result, self.sfreq / 1000)

    def make_fft_wavelets(self, freqs: Numbers,
                          real_wave_length: float = 1.) -> List[np.ndarray]:
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
        make_w = partial(self.make_fft_wavelet, real_length=real_wave_length)
        if self.interpolate:
            fft_wavelets = map(make_w, freqs)
            self.fft_wavelets = list(map(interpolate_alias, fft_wavelets))
        else:
            self.fft_wavelets = list(map(make_w, freqs))
        return self.fft_wavelets

    def formula(self, timeline: np.ndarray, freq: float) -> np.ndarray:
        ''' formula
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

    def trans_formula(self, freqs: Iterator[float],
                      freq: float = 1.) -> np.ndarray:
        ''' trans_formula
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

    def cp_trans_formula(self, freqs: Iterator[float],
                         freq: float = 1.) -> np.ndarray:
        ''' trans_formula
        The formula of Fourier Transformed Wavelet.
        Other procedures are performed by other methods.
        This is method with cupy.

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
        if freq == 0:
            raise ZeroDivisionError
        if self.mode in [WaveletMode.Reverse, WaveletMode.Twice]:
            t = self._setup_trans_shape(freq, self.real_wave_length, self.cuda)
            wavelet = ifft(self.trans_formula(t))
            half = int(wavelet.shape[0])
            start, stop = half // 2, half // 2 * 3
            total_wavelet = np.hstack((np.conj(np.flip(wavelet)), wavelet))
            wavelet = total_wavelet[start: stop]
        else:
            timeline = self._setup_waveletshape(freq, 1, zero_mean=True)
            wavelet = self.formula(timeline, freq)
        return normalize(wavelet, self.sfreq / 1000) * MNE_CONSTANT

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

    def cwt(self, wave: np.ndarray, freqs: Union[Numbers, None],
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
        if (not reuse) or (not hasattr(self, 'fft_wavelets')):
            self.make_fft_wavelets(freqs, wave.shape[0] / self.sfreq)
        pad_wave = partial(pad_to, wave_to=wave)
        wavelet = list(map(pad_wave, self.fft_wavelets))
        wavelet = cp.asarray(wavelet) if self.cuda else np.array(wavelet)
        fft_wave = cp.fft.fft(cp.asarray(wave)) if self.cuda else fft(wave)
        if self.interpolate:
            fft_wave = interpolate_alias(fft_wave, cuda=self.cuda)
        # Keep powerful even if long wave.
        fft_wave *= wave.shape[0] / self.sfreq
        if self.cuda:
            result = cp.asnumpy(cp.fft.ifft(wavelet * fft_wave))
        else:
            result = ifft(wavelet * fft_wave)
        return result

    def power(self, wave: np.ndarray, freqs: Union[Numbers, None] = None,
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
        return self.abs(wave, freqs, reuse) ** 2

    def abs(self, wave: np.ndarray, freqs: Union[Numbers, None] = None,
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
        return np.abs(self.cwt(wave, freqs, reuse))

    def plot(self, freq: float, show: bool = True) -> plt.figure:
        return plot_wavelet(self, freq, show)


def plot_wavelet(wavelet_obj: WaveletBase, freq: float,
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
    ax.plot(np.arange(0, wavelet.shape[0], 1), wavelet, label='morse')
    ax1 = fig.add_subplot(plt_num, 1, 2, projection='3d')
    ax1.scatter3D(wavelet.real, np.arange(0, wavelet.shape[0], 1),
                  wavelet.imag, label='morse')
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


def plot_tf(data: np.ndarray,
            sfreq: float = 1000,
            frange: Union[None, tuple] = None,
            trange: Union[None, tuple] = None,
            vmin: Union[float, None] = None, vmax: Union[float, None] = None,
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
    if frange is not None:
        length = frange[2] / (frange[1] - frange[0]) * data.shape[0]
        plt.yticks(np.arange(0, data.shape[0], length), np.arange(*frange))
    if trange is not None:
        plt.xticks(np.arange(0, data.shape[1], sfreq * trange[2]),
                   np.arange(*trange))
    image = ax.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.invert_yaxis()
    ax.set_aspect('auto')
    divider = make_axes_locatable(ax)
    ax_cb = divider.new_horizontal(size="2%", pad=0.05)
    fig.add_axes(ax_cb)
    plt.colorbar(image, cax=ax_cb)
    if show:
        plt.show()
    return ax
