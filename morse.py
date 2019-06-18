import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import cwt, morlet
from scipy.fftpack import fft, ifft
from scipy import sign
from math import e, pi
from typing import Union
from mpl_toolkits.mplot3d import Axes3D
from numpy import linalg, sqrt
plot: bool = True


class Morse:
    def __init__(self):
        self.r = 3
        self.b = 20
        self.l = 7
        self.a = 2 * (e * self.r / self.b) ** (self.b / self.r)
        self._setup_peak()

    def _setup_peak(self):
        self.peak = (self.b / self.r) ** (1 / self.r) / np.pi

    def beta(self, b: float) -> 'Morse':
        self.b = b
        self._setup_peak()
        return self

    def gamma(self, r: float) -> 'Morse':
        self.r = r
        self._setup_peak()
        return self

    def ln(self, l: float) -> 'Morse':
        self.peak = l
        # self.peak = self.b ** (1 / self.r)
        self.r = np.log(self.b) / np.log(self.peak)
        # self.b = 1 / np.log(l)
        # self.b = 0.05 * l ** self.r
        return self

    def make_wave(self, sfreq=1000, freq=10) -> 'Morse':
        # use 20rad
        sigma_t = 7 / (2.0 * np.pi * 1)
        self.wave_len = np.arange(0., 5. * sigma_t, 1.0 / sfreq)
        # self.wave_len = self.l / (2.0 * np.pi) 
        # self.wave_len = 1
        self._setup_peak()
        # wave_len = 11.15  # sec
        # this is not raw formula
        # length of one sample frequency

        one = 1 / sfreq * (self.b / self.r) ** (1 / self.r)
        self.one = one
        total = 5 * sigma_t * self.peak * sfreq
        #total = 100
        w = np.arange(0, total, one)
        self.w = w
        step = np.heaviside(w, w)
        # the morse wavelet
        # This is the raw formula
        self.wave = step * self.a * w ** self.b * e ** (-1 * w ** self.r)

        # inverse fft
        # This is the raw formula
        # self.morse_wave = ifft(self.wave)[:int(total / one / 2)]
        self.morse_wave = ifft(self.wave)[:int(5 * sigma_t * sfreq / freq)]
        print(self.morse_wave.shape)
        # contactnate minus value
        self.morse_wave = np.hstack((np.conj(np.flip(self.morse_wave)),
                                     self.morse_wave))
        # normalize
        self.morse_wave /= sqrt(0.5) * linalg.norm(self.morse_wave.ravel())
        return self

    def plot_wave(self):
        plt.plot(self.wave)
        plt.show()

    def plot_wavelet(self):
        plt.plot(self.morse_wave)
        plt.show()

    def wavelet(self):
        return self.morse_wave


def mne_morlet(sfreq, freqs, n_cycles=7.0, sigma=None, zero_mean=False):
    """Compute Morlet wavelets for the given frequency range.

    Parameters
    ----------
    sfreq : float
        The sampling Frequency.
    freqs : array
        frequency range of interest (1 x Frequencies)
    n_cycles: float | array of float, default 7.0
        Number of cycles. Fixed number or one per frequency.
    sigma : float, default None
        It controls the width of the wavelet ie its temporal
        resolution. If sigma is None the temporal resolution
        is adapted with the frequency like for all wavelet transform.
        The higher the frequency the shorter is the wavelet.
        If sigma is fixed the temporal resolution is fixed
        like for the short time Fourier transform and the number
        of oscillations increases with the frequency.
    zero_mean : bool, default False
        Make sure the wavelet has a mean of zero.

    Returns
    -------
    Ws : list of array
        The wavelets time series.
    """
    Ws = list()
    n_cycles = np.atleast_1d(n_cycles)

    freqs = np.array(freqs)
    if np.any(freqs <= 0):
        raise ValueError("all frequencies in 'freqs' must be "
                         "greater than 0.")

    if (n_cycles.size != 1) and (n_cycles.size != len(freqs)):
        raise ValueError("n_cycles should be fixed or defined for "
                         "each frequency.")
    for k, f in enumerate(freqs):
        if len(n_cycles) != 1:
            this_n_cycles = n_cycles[k]
        else:
            this_n_cycles = n_cycles[0]
        # fixed or scale-dependent window
        if sigma is None:
            sigma_t = this_n_cycles / (2.0 * np.pi * f)
        else:
            sigma_t = this_n_cycles / (2.0 * np.pi * sigma)
        # this scaling factor is proportional to (Tallon-Baudry 98):
        # (sigma_t*sqrt(pi))^(-1/2);
        t = np.arange(0., 5. * sigma_t, 1.0 / sfreq)
        t = np.r_[-t[::-1], t[1:]]
        oscillation = np.exp(2.0 * 1j * np.pi * f * t)
        gaussian_enveloppe = np.exp(-t ** 2 / (2.0 * sigma_t ** 2))
        if zero_mean:  # to make it zero mean
            real_offset = np.exp(- 2 * (np.pi * f * sigma_t) ** 2)
            oscillation -= real_offset
        W = oscillation * gaussian_enveloppe
        W /= sqrt(0.5) * linalg.norm(W.ravel())
        Ws.append(W)
    return Ws


n_cycles = 7
morse = Morse().beta(17.5).make_wave()
if plot:
    plt.plot(morse.wave)
    plt.show()
    plt.plot(morse.wavelet())
    plt.show()
    plt.plot(np.abs(fft(morse.wavelet())))
    plt.show()

mm = mne_morlet(1000, [10], n_cycles=n_cycles, zero_mean=True)[0]
go = mne_morlet(1000, [10], n_cycles=n_cycles)[0]
morse_wave = morse.wavelet()
morse.plot_wave()
morse.plot_wavelet()

fig = plt.figure()
ax = fig.add_subplot(211)
ax.plot(np.arange(0, morse_wave.shape[0], 1),
        morse_wave,
        label='morse')
ax.plot(np.arange(0, mm.shape[0], 1),
        mm,
        label='morlet')
ax.plot(np.arange(0, go.shape[0], 1),
        go,
        label='go')
ax1 = fig.add_subplot(212, projection='3d')
ax1.scatter3D(morse_wave.real,
              np.arange(0, morse_wave.shape[0], 1),
              morse_wave.imag,
              label='morse')
ax1.scatter3D(mm.real,
              np.arange(0, mm.shape[0], 1),
              mm.imag,
              label='morlet')
ax1.scatter3D(go.real,
              np.arange(0, mm.shape[0], 1),
              go.imag,
              label='gobar')

handler, label = ax.get_legend_handles_labels()
handler1, label1 = ax1.get_legend_handles_labels()
ax.legend(label+label1, loc='upper right')
ax.set_title('morse and morlet')
plt.show()
