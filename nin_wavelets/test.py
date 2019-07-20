import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.fftpack import ifft, fft
from typing import Union, List, Tuple
from mne.time_frequency import tfr, morlet
from nin_wavelets import Morse, MorseMNE, Morlet, enable_cupy
from .tooltip import Parallel


def test() -> None:
    morse = MorseMNE(1000, b=20)
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
    plt.plot(sin)
    plt.show()


def cwt_test(use_cuda: bool = False) -> None:
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
    plt.plot(np.abs(fft(sin)))
    plt.show()

    p = Parallel(3)
    ax1 = plt.subplot(3, 1, 1)
    ax2 = plt.subplot(3, 1, 2)
    ax3 = plt.subplot(3, 1, 3)
    ax1.invert_yaxis()
    ax2.invert_yaxis()
    ax3.invert_yaxis()

    morse = Morse()
    morse.use_cuda = use_cuda
    nin_morlet = Morlet()
    nin_morlet.use_cuda = use_cuda

    p.append(morse.power, sin, np.arange(1., 1000, 0.1))
    p.append(nin_morlet.power, sin, np.arange(1., 1000, 1))
    p.append(tfr.cwt, np.array([sin]), morlet(1000, np.arange(1, 1000, 1)))
    result_morse, result_morlet, result_mne_morlet = p.run()
    # result_morse, result_morlet = p.run()

    ax1.imshow(np.abs(result_morse), cmap='RdBu_r')
    ax2.imshow(np.abs(result_morlet), cmap='RdBu_r')
    ax3.imshow(np.abs(result_mne_morlet[0]), cmap='RdBu_r')
    ax1.invert_yaxis()
    ax2.invert_yaxis()
    ax3.invert_yaxis()
    plt.show()



def fft_wavelet_test() -> None:
    hz = 15
    r = 1
    s = 7
    morse = Morse(r=r)
    morlet = Morlet(sigma=s)
    fig = plt.figure()
    p = Parallel(4)
    p.append(morse.make_wavelets, [hz])
    p.append(morse.make_fft_wavelet, 1, 1, hz)
    p.append(morlet.make_wavelets, [hz])
    p.append(morlet.make_fft_wavelet, 1, 1, hz)
    w, a, b, c = p.run()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(w[0], label='Generalized Morse wavelet')
    ax.plot(a, label='FFTed Generalized Morse wavelet')
    ax.plot(b[0], label='Morlet wavelet')
    ax.plot(np.abs(c), label='FFTed Morlet wavelet')
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
    cwt_test(False)
