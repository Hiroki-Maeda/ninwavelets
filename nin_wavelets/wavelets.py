import numpy as np
from numpy import linalg, sqrt, e
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.fftpack import ifft, fft
from typing import Union, List, Tuple, Iterator, Iterable
from mne.time_frequency import tfr, morlet


class Morse:
    def __init__(self, sfreq: float = 1000, b: float = 17.5, r: float = 3,
                 length: float = 10, accuracy: float = 1,
                 n_jobs: int = 1) -> None:
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

    def _setup_base_waveshape(self, freq: float) -> Tuple[float, float]:
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

        one: float = 1 / freq / self.accuracy
        total: float = self.sfreq / freq
        return one, total

    def make_wavelets(self,
                      freqs: Union[List[float],
                                   range, np.ndarray]) -> List[np.ndarray]:
        '''
        Make morse wavelet.
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
        return list(map(self._ifft, freqs))

    def _make_fft_wave(self, total: float, one: float,
                       freq: float = 1) -> np.ndarray:
        '''
        Make Fourier transformed morse wavelet.

        Parameters
        ----------
        total: float | Length of wavelet.
        one: float | Sampling scale.
        '''
        self.a: float = 2 * (e * self.r / self.b) ** (self.b / self.r)
        w: np.ndarray = np.arange(0, total, one)
        step: np.ndarray = np.heaviside(w, w)
        wave: np.ndarray = (step * self.a * (w / freq) ** self.b *
                            e ** (
                                (self.b / self.r) *
                                (1 - (w / freq) ** self.r)
                            )
                            )
        return wave

    def make_fft_waves(self, total: float, one: float,
                       freqs: Iterable) -> Iterator:
        '''
        Make Fourier transformed morse wavelet.
        '''
        return map(lambda freq: self._make_fft_wave(total, one, freq),
                   freqs)

    def _ifft(self, freq: float) -> np.ndarray:
        '''
        Private method to make morse_wavelet.
        It makes wavelet, to plot or use mne.time_frequency.tfr.cwt.
        But it is not good to perform both of IFFT and FFT.
        And so, using this for mne is not good.
        Plot may be useful, so this method will not be discarded.
        '''
        one, total = self._setup_base_waveshape(freq)
        wave = self._make_fft_wave(total, one)
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
        morse_wavelet /= linalg.norm(morse_wavelet.ravel()) * sqrt(0.5)
        return morse_wavelet

    def cwt(self, wave: np.ndarray,
            freqs: Union[List[float], range, np.ndarray],
            max_freq: int = 0) -> np.ndarray:
        '''
        Run CWT.
        This method is still experimental.
        It has no error handling code now.
        '''
        one, total = self._setup_base_waveshape(freqs[0])
        wave_length = wave.shape[0]
        rate: float = wave.shape[0] / self.sfreq
        total = total / rate
        one = one / rate
        wavelet_base = self.make_fft_waves(total, one, freqs)
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
        morse = self.make_wavelets([freq])[0]
        fig = plt.figure(figsize=(6, 8))
        ax = fig.add_subplot(311)
        ax.plot(np.arange(0, morse.shape[0], 1),
                morse,
                label='morse')
        ax1 = fig.add_subplot(312, projection='3d')
        ax1.scatter3D(morse.real,
                      np.arange(0, morse.shape[0], 1),
                      morse.imag,
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
