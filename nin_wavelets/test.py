import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from typing import Any
from multiprocessing import Pool
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from mne.time_frequency import morlet
from nin_wavelets.base import interpolate_alias
from nin_wavelets import Morse, MorseMNE, Morlet, WaveletMode, Haar, plot_tf
import gc


def make_example(length: float = 3) -> np.ndarray:
    freq: float = 60
    time: np.ndarray = np.arange(0, length, 0.001)
    sin = np.array(np.sin(time * freq * 2 * np.pi) +
                   np.sin(time * 160 * 2 * np.pi) * np.sin(time * np.pi) +
                   np.sin(np.pad(np.arange(0, length / 2, 0.001),
                                 [int(length * 250), int(length * 250)],
                                 'constant') *
                          300 * 2 * np.pi)
                   )
    return sin


def test() -> None:
    morse = Morse(1000, 17.5, 3)
    freq = 60
    time = np.arange(0, 3, 0.001)
    sin = np.array(np.sin(time * freq * 2 * np.pi))
    result = morse.power(sin, range(1, 100))
    plt.imshow(result, cmap='RdBu_r')
    plt.gca().invert_yaxis()
    plt.title('CWT of 60Hz sin wave')
    plt.show()


def test3d() -> None:
    go = morlet(1000, [10])[0]
    mm = morlet(1000, [10], zero_mean=True)[0]
    morse_obj = Morse(1000, 17.5, 3)
    morse = morse_obj.make_wavelets([10])[0]
    nin_morlet = Morlet(1000).make_wavelets([10])[0]
    fig = plt.figure()
    ax = fig.add_subplot(211)
    half_morse = morse.shape[0] / 2
    half_mm = mm.shape[0] / 2
    ax.plot(np.arange(-half_morse, half_morse, 1),
            morse,
            label='Morse Wavelet')
    ax.plot(np.arange(-half_morse, half_morse, 1),
            nin_morlet,
            label='Morlet Wavelet')
    ax.plot(np.arange(-half_morse, half_morse, 1),
            morse.imag,
            label='Morse Imag')
    ax.plot(np.arange(-half_mm, half_mm, 1),
            mm,
            label='Morlet')
    ax.plot(np.arange(-half_mm, half_mm, 1),
            mm.imag,
            label='Morlet imag')
    ax.plot(np.arange(-half_mm, half_mm, 1),
            go,
            label='Gabor Wavelet')
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


def plot_sin_fft() -> None:
    freq = 60
    time = np.arange(0, 0.3, 0.001)
    sin = np.array(np.sin(time * freq * 2 * np.pi))
    sin = np.hstack((sin, np.zeros(100)))
    plt.plot(sin)
    plt.show()
    plt.plot(np.abs(fft(sin)))
    plt.show()
    plt.plot(ifft(fft(sin)))
    plt.plot(ifft(interpolate_alias(fft(sin))))
    plt.show()


def cwt_test(interpolate: bool = True) -> None:
    sin = make_example(2)
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)
    ax1.invert_yaxis()
    ax2.invert_yaxis()

    morse = Morse(interpolate=interpolate)
    nin_morlet = Morlet(interpolate=interpolate)
    nin_morlet.mode = WaveletMode.Both

    result_morse = morse.power(sin, np.arange(1., 1000, 1))
    result_morlet = nin_morlet.power(sin, np.arange(1., 1000, 1))

    vmax = 0.03
    ax1.imshow(np.abs(result_morse), cmap='RdBu_r', vmax=vmax)
    ax2.imshow(np.abs(result_morlet), cmap='RdBu_r', vmax=vmax)
    ax1.invert_yaxis()
    ax2.invert_yaxis()
    ax1.set_title('Morse')
    ax2.set_title('Morlet')
    plt.show()
    result_morse = morse.power(sin, reuse=True)
    plot_tf(result_morse)
    plt.show()


def fft_wavelet_test() -> None:
    hz = 10.
    r = 3
    b = 17.5
    s = 7
    morse = Morse(r=r, b=b)
    morlet = Morlet(sigma=s)
    fig = plt.figure()
    w = morse.make_wavelet(hz)
    a = morse.make_fft_wavelet(hz)
    b = morlet.make_wavelet(hz)
    c = morlet.make_fft_wavelet(hz)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(w, label='Generalized Morse wavelet')
    ax.plot(a, label='FFTed Generalized Morse wavelet')
    ax.plot(b, label='Morlet wavelet')
    ax.plot(b.imag, label='Morlet wavelet')
    # ax.plot(np.abs(c), label='FFTed Morlet wavelet abs')
    ax.plot(np.abs(c.real), label='FFTed Morlet wavelet')
    ax.plot(c.imag, label='imag of FFTed Morlet wavelet')
    handler, label = ax.get_legend_handles_labels()
    ax.legend(label, loc='upper right')
    plt.show()


if __name__ == '__main__':
    # enable_cupy()
    print('Test Run')
    # plot_sin_fft()
    # test()
    # test3d()
    fft_wavelet_test()
    cwt_test(True)
