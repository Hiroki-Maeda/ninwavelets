import numpy as np
from numpy import linalg, sqrt, e
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.fftpack import ifft, fft
from typing import Union, List, Tuple
from mne.time_frequency import tfr, morlet


class Morse:
    def __init__(self, sfreq: float = 1000, b: float = 17.5, r: float = 3,
                 length: float = 10, accuracy: float = 1) -> None:
        '''
        Generator of Generalized Morse Wavelets.
        It is compatible with mne-python.
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
            Because, Morse Wavelet needs Fourier Transform,
            length of wavelet changes but it's tiring to detect. :(
            Low frequency causes bad wave.
            Please check wave by Morse.plot(freq) before use it.
            If wave is bad, large accuracy can help you.(But needs cpu power)
        length: float | Length paramater.
            Too long wavelet causes slow calculation.
            This param is cutting threshould of wave.
            Peak wave * length is the length of wavelet.

        Returns
        -------
        Morse instance its self.
        '''
        self._peak: float
        self.r: float = r
        self.b: float = b
        self.length: float = length
        self.accuracy: float = accuracy
        self.sfreq: float = sfreq
        self._setup_peak()

    def _setup_peak(self) -> None:
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

    def _setup_waveshape(self, freq: float) -> Tuple[float, float]:
        '''
        Set gamma value of MorseWavelet.
        Good value may be 3.

        Parameters
        ----------
        freq: float | Frequency

        Returns
        -------
        Tuple[float, float]: (one, total)
        '''

        one = 1 / freq / self.accuracy
        total = self.sfreq / freq
        return one, total

    def wavelet(self, freq: float = 10) -> np.ndarray:
        '''
        Make morse wavelet.
        It returnes one freq wavelet only, and so it may be useless.

        Parameters
        ----------
        freq: float | Frequency. If frequency is too small,
            it returnes bad wave easily.
            For example, sfreq=1000, freq=3 it returnes bad wave.
            If you want good wave, you must set large accuracy, and length
            when you make this instance.

        Returns
        -------
        MorseWavelet: np.ndarray
        '''
        self._setup_peak()
        one = 1 / freq / self.accuracy
        total = self.sfreq / freq
        return self._ifft(total, one, freq)

    def wavelets(self,
                 freqs: Union[List[float], range, np.ndarray]) -> np.ndarray:
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
        return map(self.wavelet, freqs)

    def fft_wave(self, total: float, one: float,
                 freq: float = 1) -> np.ndarray:
        '''
        Make Fourier transformed morse wavelet.
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

    def fft_waves(self, total: float, one: float,
                  freqs: List[float]) -> np.ndarray:
        '''
        Make Fourier transformed morse wavelet.
        '''
        return map(lambda freq: self.fft_wave(total, one, freq),
                   freqs)

    def _ifft(self, total: float, one: float, freq: float) -> np.ndarray:
        '''
        Private method to make morse_wavelet.
        It makes wavelet, to use mne.time_frequency.tfr.cwt.
        But it is not good to perform both of IFFT and FFT.
        Because, wave form changes.
        I am going to write method not to use mne.
        '''
        wave = self.fft_wave(total, one)
        morse_wavelet: np.ndarray = ifft(wave)
        half = int(morse_wavelet.shape[0])
        band = int(half / 2 / freq * self.length)
        start: int = half - band if band < half // 2 else half // 2
        stop: int = half + band if band < half // 2 else half // 2 * 3
        total_wavelet: np.ndarray = np.hstack((np.conj(np.flip(morse_wavelet)),
                                               morse_wavelet))
        morse_wavelet = total_wavelet[start: stop]
        # normalize
        morse_wavelet /= sqrt(0.5) * linalg.norm(morse_wavelet.ravel())
        return morse_wavelet

    def cwt(self, wave: np.ndarray,
            freqs: Union[List[float], range, np.ndarray]) -> np.ndarray:
        pass

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
        morse = self.wavelet(freq)
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
                        labeltop=False, bottom=False,
                        left=False, right=False, top=False)
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
                       list(self.wavelets(range(1, 100))),
                       use_fft=use_fft,
                       mode=mode, decim=decim).mean(axis=0)


def test() -> None:
    morse = MorseMNE(1000).beta(20)
    freq = 60
    time = np.arange(0, 0.3, 0.001)
    sin = np.array([np.sin(time * freq * 2 * np.pi)])
    result = morse.power(sin, range(1, 100))
    plt.imshow(result, cmap='RdBu_r')
    plt.gca().invert_yaxis()
    plt.title('CWT of 60Hz sin wave')
    plt.show()


def test3d() -> None:
    go = morlet(1000, [10])[0]
    mm = morlet(1000, [10], zero_mean=True)[0]
    morse = MorseMNE(1000, 17.5, 3).wavelet(10)
    fig = plt.figure()
    ax = fig.add_subplot(211)
    half_morse = morse.shape[0] / 2
    half_mm = mm.shape[0] / 2
    ax.plot(np.arange(-half_morse, half_morse, 1),
            morse,
            label='morse')
    ax.plot(np.arange(-half_mm, half_mm, 1),
            mm,
            label='morlet')
    ax.plot(np.arange(-half_mm, half_mm, 1),
            go,
            label='go')
    ax1 = fig.add_subplot(212, projection='3d')
    ax1.scatter3D(morse.real,
                  np.arange(-half_morse, half_morse, 1),
                  morse.imag,
                  label='morse')
    ax1.scatter3D(mm.real,
                  np.arange(-half_mm, half_mm, 1),
                  mm.imag,
                  label='morlet')
    ax1.scatter3D(go.real,
                  np.arange(-half_mm, half_mm, 1),
                  go.imag,
                  label='gobar')
    handler, label = ax.get_legend_handles_labels()
    handler1, label1 = ax1.get_legend_handles_labels()
    ax.legend(label+label1, loc='upper right')
    ax.set_title('morse and morlet')
    plt.show()


if __name__ == '__main__':
    print('Test Run')
    test()
    Morse().plot(10)
