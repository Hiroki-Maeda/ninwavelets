import numpy as np
from numpy import linalg, sqrt, e
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.fftpack import ifft, fft
from typing import Union, List, Tuple
from mne.time_frequency import tfr, morlet
from wavelets import Morse, MorseMNE


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
    morse = Morse(1000, 17.5, 3).make_wavelets([10])[0]
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


def plot_sin_fft() -> None:
    freq = 60
    time = np.arange(0, 0.3, 0.001)
    sin = np.array(np.sin(time * freq * 2 * np.pi))
    # plt.plot(sin)
    plt.plot(sin)
    plt.show()


def cwt_test() -> None:
    freq: float = 60
    length = 3
    time: np.ndarray = np.arange(0, length, 0.001)
    sin = np.array(np.sin(time * freq * 2 * np.pi) +
                   np.sin(time * 160 * 2 * np.pi) * np.sin(time * np.pi) +
                   np.sin(np.pad(np.arange(0, length / 2, 0.001),
                                 [int(length * 250), int(length * 250)],
                                 'constant') *
                          300 * 2 * np.pi)
                   )
    plt.plot(sin)
    plt.show()
    morse = Morse()
    result = morse.cwt(sin, np.arange(1, 1000, 1))
    plt.imshow(np.abs(result), cmap='RdBu_r')
    plt.show()
    result = morse.power(sin, np.arange(1, 1000, 1))
    plt.imshow(np.abs(result), cmap='RdBu_r')
    plt.show()


def fft_wavelet_test() -> None:
    w = Morse().make_wavelets([10])[0]
    plt.plot(w)
    plt.show()


if __name__ == '__main__':
    print('Test Run')
    plot_sin_fft()
    test()
    test3d()
    fft_wavelet_test()
    cwt_test()
