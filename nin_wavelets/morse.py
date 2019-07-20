from .wavelets import WaveletBase, WaveletMode
from typing import Tuple, Union, List, Iterable, Iterator
from scipy.fftpack import fft, ifft
import numpy as np
from mne.time_frequency import tfr
from enum import Enum


class Morse(WaveletBase):

    def __init__(self, sfreq: float = 1000, b: float = 17.5, r: float = 3,
                 length: float = 10, accuracy: float = 1) -> None:
        '''
        Generator of Generalized Morse Wavelets.
        Example.
        >>> morse = Morse(1000).beta(20)
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
        super(Morse, self).__init__(sfreq)
        self.r: float = r
        self.b: float = b
        self.length: float = length
        self.accuracy: float = accuracy
        self.sfreq: float = sfreq
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

    def trans_wavelet_formula(self, w: np.ndarray,
                              freq: float = 1.) -> np.ndarray:
        '''
        Make Fourier transformed morse wavelet.
        '''
        w = w / freq
        step: np.ndarray = np.heaviside(w, w)
        wave: np.ndarray = 2 * (step * (w ** self.b) *
                                np.e ** ((self.b / self.r) *
                                         (1 - w ** self.r)
                                         )) / np.pi
        return wave


class Morlet(WaveletBase):
    '''
    Base class of wavelets.
    '''
    def __init__(self, sfreq: float = 1000, sigma: float = 7.,
                 accuracy: float = 1.) -> None:
        self.mode = WaveletMode.Normal
        # self.mode = WaveletMode.Both
        self.length = 10
        self.sfreq = sfreq
        self.sigma = sigma
        self.accuracy = accuracy
        self.c = (1 + np.e ** (-self.sigma ** 2 / 2) -
                  2 * np.e ** (-3 / 4 * self.sigma ** 2)) ** (-1/2)
        self.k = np.e ** (-self.sigma ** 2 / 2)

    def trans_wavelet_formula(self, timeline: np.ndarray,
                              freq: float = 1) -> np.ndarray:
        timeline = timeline / freq * self.peak_freq(freq)
        return (self.c * np.pi ** (-1/4) *
                (np.e**(-(self.sigma-timeline)**2/2) -
                 self.k * np.e ** (-timeline**2/2)))

    def wavelet_formula(self, timeline: np.ndarray,
                        freq: float = 1) -> np.ndarray:
        return (self.c * np.pi ** (-1 / 4)
                * np.e ** (-timeline ** 2 / 2)
                * (np.e ** (self.sigma * 1j * timeline - self.k)))

    def peak_freq(self, freq: float) -> float:
        return self.sigma / (1. - np.e ** (-self.sigma * freq))


class MorseMNE(Morse):
    '''
    MorseWavelets for mne
    '''

    def __init__(self, sfreq: float = 1000, b: float = 17.5, r: float = 3,
                 length: float = 10, accuracy: float = 1) -> None:
        super(MorseMNE, self).__init__(sfreq, b, r, length, accuracy)

    def cwt(self, wave: np.ndarray,
            freqs: Union[List[float], range, np.ndarray],
            use_fft: bool = True, mode: str = 'same',
            decim: float = 1) -> np.ndarray:
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
