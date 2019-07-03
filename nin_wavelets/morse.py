from .wavelets import WaveletBase
from typing import Tuple, Union, List, Iterable, Iterator
import numpy as np
from scipy.fftpack import ifft
from mne.time_frequency import tfr


class Morse(WaveletBase):
    def __init__(self, sfreq: float = 1000, b: float = 17.5, r: float = 3,
                 length: float = 10, accuracy: float = 1) -> None:
        '''
        Generator of Generalized Morse Wavelets.
        It is compatible with mne-python.(But not recommended.)
        It generates argument Ws for mne.time_frequency.tfr.cwt.
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
        self._peak: float
        self.r: float = r
        self.b: float = b
        self.length: float = length
        self.accuracy: float = accuracy
        self.sfreq: float = sfreq
        self._setup_peak()

    def _setup_peak(self) -> None:
        '''
        Set up peak frequency.
        It is private method.
        '''
        self.peak: float = (self.b / self.r) ** (1 / self.r)

    def beta(self, b: float) -> 'Morse':
        '''
        Set gamma value of MorseWavelet.
        If it is 17.5, MorseWavelets may resembles MorseWavelet with sigma 7.

        Parameters
        ----------
        r: float | gamma value

        Returns
        -------
        Morse instance its self.
        '''
        self.b = b
        self._setup_peak()
        return self

    def gamma(self, r: float) -> 'Morse':
        '''
        Set gamma value of MorseWavelet.
        Good value may be 3.

        Parameters
        ----------
        r: float | gamma value

        Returns
        -------
        Morse instance its self.
        '''
        self.r = r
        self._setup_peak()
        return self

    def _make_fft_wavelet(self, total: float, one: float,
                          freq: float = 1) -> np.ndarray:
        '''
        Make Fourier transformed morse wavelet.

        Parameters
        ----------
        total: float | Length of wavelet.
        one: float | Sampling scale.
        '''
        self.a: float = 2 * (np.e * self.r / self.b) ** (self.b / self.r)
        w: np.ndarray = np.arange(0, total, one)
        step: np.ndarray = np.heaviside(w, w)
        wave: np.ndarray = (step * self.a * (w / freq) ** self.b *
                            np.e ** (
                                (self.b / self.r) *
                                (1 - (w / freq) ** self.r)
                            )
                            )
        return wave

    def _make_wavelet(self, freq: float) -> np.ndarray:
        '''
        Private method to make morse_wavelet.
        It makes wavelet, to plot or use mne.time_frequency.tfr.cwt.
        But it is not good to perform both of IFFT and FFT.
        And so, using this for mne is not good.
        Plot may be useful, so this method will not be discarded.
        '''
        one, total = self._setup_base_waveshape(freq)
        wave = self._make_fft_wavelet(total, one)
        morse_wavelet: np.ndarray = ifft(wave)
        half = int(morse_wavelet.shape[0])
        band = int(half / 2 / freq * self.length)
        start: int = half - band if band < half // 2 else half // 2
        stop: int = half + band if band < half // 2 else half // 2 * 3
        # cut side of wavelets and contactnate
        total_wavelet: np.ndarray = np.hstack((np.conj(np.flip(morse_wavelet)),
                                               morse_wavelet))
        morse_wavelet = total_wavelet[start: stop]
        # normalize
        morse_wavelet /= np.linalg.norm(morse_wavelet.ravel()) * np.sqrt(0.5)
        return morse_wavelet


class MorseMNE(Morse):
    '''
    MorseWavelets for mne
    '''
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
