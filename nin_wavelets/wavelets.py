from .base import WaveletBase, WaveletMode
from typing import Union, List
import numpy as np


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
    accuracy: float | Accurancy paramater.
        It does not make sence when you use fft only.
        Because, Morse Wavelet needs Inverse Fourier Transform,
        length of wavelet changes but it is tiring to detect. :(
        If you use ifft, low frequency causes bad wave.
        Please check wave by Morse.plot(freq) before use it.
        If wave is bad, large accuracy can help you.(But needs cpu power)
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
                 accuracy: float = 1, real_wave_length: float = 1.,
                 interpolate: bool = False) -> None:
        super(Morse, self).__init__(sfreq, accuracy,
                                    real_wave_length, interpolate)
        self.r: float = r
        self.b: float = b
        self.mode = WaveletMode.Reverse
        self.help = '''This is inverse Fourier transformed MorseWavelet.
                    Originally, Generalized Morse wavelet is
                    Frourier transformed wave.
                    It should be used as it is Fourier transformed data.
                    But, you can use it in the same way as'
                    MorletWavelet by IFFT.
                    If wave continues to side of the window, wave is bad.
                    Please set larger value to param
                    accuracy" and "length"
                    It becomes bad easily when frequency is low.'''

    def trans_wavelet_formula(self, freqs: np.ndarray,
                              freq: float = 1.) -> np.ndarray:
        '''
        Make Fourier transformed morse wavelet.
        '''
        freqs = freqs / freq
        step: np.ndarray = np.heaviside(freqs, freqs)
        wave: np.ndarray = 2 * (step * (freqs ** self.b) *
                                np.e ** ((self.b / self.r) *
                                         (1 - freqs ** self.r)
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
    sfreq: float | Sampling frequency.
        This behaves like sfreq of mne-python.
    sigma: float | sigma value
    accuracy: float | Accurancy paramater.
        It does not make sence when you use fft only.
        Because, Morse Wavelet needs Inverse Fourier Transform,
        length of wavelet changes but it is tiring to detect. :(
        If you use ifft, low frequency causes bad wave.
        Please check wave by Morse.plot(freq) before use it.
        If wave is bad, large accuracy can help you.(But needs cpu power)
    length: float | Length of wavelet.
        It does not make sence when you use fft only.
        Too long wavelet causes slow calculation.
        This param is cutting threshould of wavelets.
        Peak wave * length is the length of wavelet.

    Returns
    -------
    As constructor, Morse instance its self.
    '''
    def __init__(self, sfreq: float = 1000, sigma: float = 7.,
                 accuracy: float = 1., real_wave_length: float = 1.,
                 interpolate: bool = False) -> None:
        super(Morlet, self).__init__(sfreq, accuracy,
                                     real_wave_length, interpolate)
        self.mode = WaveletMode.Normal
        # self.mode = WaveletMode.Both
        self.sigma = sigma
        self.c = (1 + np.e ** (-self.sigma ** 2 / 2) -
                  2 * np.e ** (-3 / 4 * self.sigma ** 2)) ** (-1/2)
        self.k = np.e ** (-self.sigma ** 2 / 2)

    def trans_wavelet_formula(self, freqs: np.ndarray,
                              freq: float = 1) -> np.ndarray:
        freqs = freqs / freq * self.peak_freq(freq)
        return (self.c * np.pi ** (-1/4) *
                (np.e**(-(self.sigma-freqs)**2/2) -
                 self.k * np.e ** (-freqs**2/2)))

    def wavelet_formula(self, timeline: np.ndarray,
                        freq: float = 1) -> np.ndarray:
        return (self.c * np.pi ** (-1 / 4)
                * np.e ** (-timeline ** 2 / 2)
                * (np.e ** (self.sigma * 1j * timeline - self.k)))

    def peak_freq(self, freq: float) -> float:
        return self.sigma / (1. - np.e ** (-self.sigma * freq))


class MorseMNE(Morse):
    '''
    MorseWavelets for mne.
    It uses GMW with mne function.
    But, it use iFFT and FFT to no purpose.
    This ugly class is disgusting and depricated.
    '''

    def __init__(self, sfreq: float = 1000, b: float = 17.5, r: float = 3,
                 accuracy: float = 1., real_wave_length: float = 1.,
                 interpolate: bool = False) -> None:
        super(MorseMNE, self).__init__(sfreq, accuracy,
                                       real_wave_length, interpolate)
        self.r: float = r
        self.b: float = b
        self.mode = WaveletMode.Reverse
        self.help = '''This is inverse Fourier transformed MorseWavelet.
                    Originally, Generalized Morse wavelet is
                    Frourier transformed wave.
                    It should be used as it is Fourier transformed data.
                    But, you can use it in the same way as'
                    MorletWavelet by IFFT.
                    If wave continues to side of the window, wave is bad.
                    Please set larger value to param
                    accuracy" and "length"
                    It becomes bad easily when frequency is low.'''


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


class Haar(WaveletBase):
    def __init__(self, sfreq: float = 1000,
                 accuracy: float = 1., real_wave_length: float = 1.,
                 interpolate: bool = False) -> None:
        super(Haar, self).__init__(sfreq, accuracy,
                                   real_wave_length, interpolate)
        self.mode = WaveletMode.Normal

    def wavelet_formula(self, timeline: np.ndarray,
                        freq: float = 1) -> np.ndarray:
        for key, value in enumerate(timeline):
            if (0. < value) and (value <= 1.):
                timeline[key] = 1.
            elif (-1. < value) and (value <= 0.):
                timeline[key] = -1.
            else:
                timeline[key] = 0.
        return timeline
