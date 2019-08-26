from .base import WaveletBase, WaveletMode
from typing import Union, List
import numpy as np
import cupy as cp


class Morse(WaveletBase):
    '''
    Generator of Generalized Morse Wavelets.
    Example.
    >>> morse = Morse(1000, r=3., b=17.5)
    >>> freq = 60
    >>> time = np.arange(0, 0.3, 0.001)
    >>> sin = np.array([np.sin(time * freq * 2 * np.pi)])
    >>> result = morse.power(sin, range(1, 100))
    >>> plt.imshow(result, cmap='RdBu_r')
    >>> plt.gca().invert_yaxis()
    >>> plt.title('CWT of 60Hz sin wave')
    >>> plt.show()

    Parameters
    ----------
    sfreq: float | Sampling frequency.
        This behaves like sfreq of mne-python.
    b: float | beta value
    r: float | gamma value. 3 may be good value.
    length: float | Length of wavelet.
        It does not make sence when you use fft only.
        Too long wavelet causes slow calculation.
        This param is cutting threshould of wavelets.
        Peak wave * length is the length of wavelet.

    Returns
    -------
    As constructor, Morse instance its self.
    '''

    def __init__(self, sfreq: float = 1000, b: float = 17.5, r: float = 3,
                 real_wave_length: float = 1.,
                 interpolate: bool = False, cuda: bool = False) -> None:
        super(Morse, self).__init__(sfreq, real_wave_length,
                                    interpolate, cuda)
        self.r: float = r
        self.b: float = b
        self.mode = WaveletMode.Reverse
        self.help = '''This is inverse Fourier transformed MorseWavelet.
Originally, Generalized Morse wavelet is
Frourier transformed wave.
It should be used as it is Fourier transformed data.
But, you can use it in the same way as'
MorletWavelet by IFFT.'''

    def cp_trans_formula(self, freqs: cp.ndarray,
                                 freq: float = 1.) -> cp.ndarray:
        np_freqs = cp.asnumpy(freqs)
        step = cp.asarray(np.heaviside(np_freqs, np_freqs))
        freqs = cp.asarray(freqs) / freq
        wave = 2. * (step * freqs ** self.b *
                     cp.exp((self.b / self.r) *
                            (1.
                             - freqs ** self.r)
                            )) / cp.pi
        return wave

    def trans_formula(self, freqs: np.ndarray,
                              freq: float = 1.) -> np.ndarray:
        '''
        Make Fourier transformed morse wavelet.
        '''
        freqs = freqs / freq
        step = np.heaviside(freqs, freqs)
        wave = 2. * (step * np.float_power(freqs, self.b) *
                     np.exp((self.b / self.r) *
                            (1.
                             - np.float_power(freqs, self.r))
                            )) / np.pi
        return wave


class Morlet(WaveletBase):
    '''
    Morlet Wavelets.
    Example.
    >>> morse = Morse(1000, sigma=7.)
    >>> freq = 60
    >>> time = np.arange(0, 0.3, 0.001)
    >>> sin = np.array([np.sin(time * freq * 2 * np.pi)])
    >>> result = morse.power(sin, range(1, 100))
    >>> plt.imshow(result, cmap='RdBu_r')
    >>> plt.gca().invert_yaxis()
    >>> plt.title('CWT of 60Hz sin wave')
    >>> plt.show()

    Parameters
    ----------
    sfreq: float
        Sampling frequency.
        This behaves like sfreq of mne-python.
    sigma: float
        sigma value
    length: float
        Length of wavelet.
        It does not make sence when you use fft only.
        Too long wavelet causes slow calculation.
        This param is cutting threshould of wavelets.
        Peak wave * length is the length of wavelet.

    Returns
    -------
    As constructor, Morse instance its self.
    '''

    def __init__(self, sfreq: float = 1000, sigma: float = 7.,
                 real_wave_length: float = 1.,
                 gabor: bool = False, interpolate: bool = False,
                 cuda: bool = False) -> None:
        super(Morlet, self).__init__(sfreq, real_wave_length,
                                     interpolate, cuda)
        self.mode = WaveletMode.Both
        self.sigma = sigma
        self.c = np.float_power(1 +
                                np.exp(-np.float_power(self.sigma, 2) / 2)
                                - 2 * np.exp(-3 / 4
                                             * np.float_power(self.sigma, 2)),
                                -1/2)
        self.k = 0 if gabor else np.exp(-np.float_power(self.sigma, 2) / 2)

    def cp_trans_formula(self, freqs: cp.ndarray,
                                 freq: float = 1.) -> cp.ndarray:
        freqs = freqs / freq * self.peak_freq(freq)
        result = (self.c * cp.pi ** (-1/4) *
                  (cp.exp(-cp.square(self.sigma-freqs) / 2) -
                   self.k * cp.exp(-cp.square(freqs) / 2)))
        return result

    def trans_formula(self, freqs: np.ndarray, freq: float = 1) -> np.ndarray:
        freqs = freqs / freq * self.peak_freq(freq)
        return (self.c * np.float_power(np.pi, (-1/4)) *
                (np.exp(-np.square(self.sigma-freqs) / 2) -
                 self.k * np.exp(-np.square(freqs) / 2)))

    def formula(self, timeline: np.ndarray, freq: float = 1) -> np.ndarray:
        return (self.c * np.float_power(np.pi, (-1 / 4))
                * np.exp(-np.square(timeline) / 2)
                * (np.exp(self.sigma * 1j * timeline) - self.k))

    def peak_freq(self, freq: float) -> float:
        return self.sigma / (1. - np.exp(-self.sigma * freq))


class MorseMNE(Morse):
    '''
    MorseWavelets for mne.
    It uses GMW with mne function.
    But, it use iFFT and FFT to no purpose.
    This ugly class is disgusting and depricated.
    '''

    def __init__(self, sfreq: float = 1000, b: float = 17.5, r: float = 3,
                 real_wave_length: float = 1.,
                 interpolate: bool = False, cuda: bool = False) -> None:
        super(MorseMNE, self).__init__(sfreq, real_wave_length,
                                       interpolate, cuda)
        self.r: float = r
        self.b: float = b
        self.mode = WaveletMode.Reverse
        self.help = '''This is inverse Fourier transformed MorseWavelet.
Originally, Generalized Morse wavelet is
Frourier transformed wave.
It should be used as it is Fourier transformed data.
But, you can use it in the same way as'
MorletWavelet by IFFT.'''

    def cwt(self, wave: np.ndarray,
            freqs: Union[List[float], range, np.ndarray],
            use_fft: bool = True, mode: str = 'same',
            decim: float = 1) -> np.ndarray:
        from mne.time_frequency import tfr
        '''
        Run cwt of mne-python.
        Because of use of IFFT before FFT to the same wave,
        this ugly method is disgusting.

        Parameters
        ----------
        freqs: float | Frequencies. Before use this, please run plot.

        Returns
        -------
        Result of cwt. Complex np.ndarray.
        '''
        return tfr.cwt(wave,
                       list(self.make_wavelets(range(1, 100))),
                       use_fft=use_fft,
                       mode=mode, decim=decim).mean(axis=0)


class MexicanHat(WaveletBase):
    '''
    Generator of MexicanHat Wavelets.

    Parameters
    ----------
    sfreq: float | Sampling frequency.
        This behaves like sfreq of mne-python.
    sigma: float | sigma value
    length: float | Length of wavelet.

    Returns
    -------
    As constructor, MexicanHat instance its self.
    '''

    def __init__(self, sfreq: float = 1000, sigma: float = 7,
                 real_wave_length: float = 1.,
                 interpolate: bool = False, cuda: bool = False) -> None:
        super(MexicanHat, self).__init__(sfreq, real_wave_length,
                                         interpolate, cuda)
        self.sigma: float = sigma
        self.mode = WaveletMode.Normal
        self.help = ''

    def formula(self, tc: np.ndarray, freq: float = 1) -> np.ndarray:
        return ((1 - np.power(tc / self.sigma, 2))
                * np.exp(-np.square(tc) / np.square(self.sigma) / 2))

    def cp_formula(self, tc: np.ndarray, freq: float = 1) -> np.ndarray:
        return ((1 - cp.power(tc / self.sigma, 2))
                * cp.exp(-cp.square(tc) / cp.square(self.sigma) / 2))

    def peak_freq(self, freq: float) -> float:
        return np.sqrt(6) / np.pi / np.pi


class Shannon(WaveletBase):
    '''
    Generator of MexicanHat Wavelets.

    Parameters
    ----------
    sfreq: float | Sampling frequency.
        This behaves like sfreq of mne-python.
    sigma: float | sigma value
    length: float | Length of wavelet.

    Returns
    -------
    As constructor, MexicanHat instance its self.
    '''

    def __init__(self, sfreq: float = 1000, sigma: float = 7,
                 real_wave_length: float = 1.,
                 interpolate: bool = False, cuda: bool = False) -> None:
        super(Shannon, self).__init__(sfreq, real_wave_length,
                                      interpolate, cuda)
        self.sigma: float = sigma
        self.mode = WaveletMode.Normal
        self.help = ''

    def formula(self, tc: np.ndarray, freq: float = 1) -> np.ndarray:
        tc /= freq
        return (np.sinc(self.sigma * tc) * np.exp(2 * np.pi * 1j * tc))

    def cp_formula(self, tc: np.ndarray, freq: float = 1) -> np.ndarray:
        tc /= freq
        return (np.sinc(self.sigma * tc) * np.exp(2 * np.pi * 1j * tc))

    def peak_freq(self, freq: float) -> float:
        return np.pi / 2


class Haar(WaveletBase):
    def __init__(self, sfreq: float = 1000,
                 real_wave_length: float = 1.,
                 interpolate: bool = False) -> None:
        super(Haar, self).__init__(sfreq, real_wave_length, interpolate)
        self.mode = WaveletMode.Normal

    def formula(self, timeline: np.ndarray,
                        freq: float = 1) -> np.ndarray:
        for key, value in enumerate(timeline):
            if (0. < value) and (value <= 1.):
                timeline[key] = 1.
            elif (-1. < value) and (value <= 0.):
                timeline[key] = -1.
            else:
                timeline[key] = 0.
        return timeline
