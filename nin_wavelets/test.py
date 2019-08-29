import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from typing import Any
from multiprocessing import Pool
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from mne.time_frequency import morlet
from nin_wavelets.base import interpolate_alias
from nin_wavelets import Morse, MorseMNE, Morlet, WaveletMode, Haar, plot_tf, MexicanHat, Shannon
from mne.io import Raw
import gc
from sys import argv


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
    morse = morse_obj.make_wavelet(10)
    nm = Morlet(1000)
#    nm.mode = WaveletMode.Normal
    nin_morlet = nm.make_wavelet(10)

    half_morse = morse.shape[0] / 2
    morse_time = np.arange(-half_morse, half_morse, 1)
    half_mm = mm.shape[0] / 2
    morlet_time = np.arange(-half_mm, half_mm, 1)
    fig = plt.figure()
    ax = fig.add_subplot(211)

    ax.plot(morse_time, morse, label='Morse Wavelet')
    ax.plot(morse_time, nin_morlet, label='Morlet Wavelet')
    ax.plot(morse_time, morse.imag, label='Morse Imag')
    ax.plot(morlet_time, mm, label='MNE Morlet')
    ax.plot(morlet_time, mm.imag, label='MNE Morlet imag')
    ax.plot(morlet_time, go, label='Gabor Wavelet')

    ax1 = fig.add_subplot(212, projection='3d')
    ax1.scatter3D(morse.real, morse_time, morse.imag, label='morse')
    ax1.scatter3D(mm.real, morlet_time, mm.imag, label='MNE morlet')
    ax1.scatter3D(go.real, morlet_time, go.imag, label='gobar')
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


def simple_plot_test() -> None:
    morse = Morse()
    morse.plot(10)


def cwt_test(interpolate: bool = True, cuda: bool = False) -> None:
    sin = make_example(10)
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)
    ax1.invert_yaxis()
    ax2.invert_yaxis()

    morse = Morse(interpolate=interpolate, cuda=cuda)
    nin_morlet = Morlet(interpolate=interpolate, cuda=cuda, sfreq=500)
    nin_morlet.mode = WaveletMode.Both

    result_morse = morse.power(sin, range(1, 1000))
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


def other_wavelet_test() -> None:
    hz = 10
    s = 7
    mexcan = MexicanHat().make_wavelet(hz)
    shannon = Shannon().make_wavelet(hz)
    morlet = Morlet(sigma=s).make_wavelet(hz)
    plt.plot(mexcan)
    plt.plot(shannon)
    plt.plot(morlet)
    plt.show()
    plt.plot(np.abs(fft(shannon)))
    plt.plot(np.abs(fft(morlet)))
    plt.show()


def fft_wavelet_test() -> None:
    hz = 10.
    r = 3
    b = 17.5
    s = 7
    morse = Morse(r=r, b=b)
    morlet = Morlet(sigma=s, sfreq=1000)
    normal_morlet = Morlet(sigma=s, sfreq=1000)
    normal_morlet.mode = WaveletMode.Normal
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
    ax.plot(np.abs(c.real), label='FFTed Morlet wavelet')
    ax.plot(c.imag, label='imag of FFTed Morlet wavelet')
    ax.plot(normal_morlet.make_wavelet(hz), label='Morlet Wavelet Normal Mode')
    handler, label = ax.get_legend_handles_labels()
    ax.legend(label, loc='upper right')
    plt.show()


def eeg() -> None:
    '''
    This test code reads my eeg.
    I am not sure whether I can open my eeg.
    My boss may says "You shoulnt!"
    If you have your own eeg, why dont you process your eeg?
    '''
    raw = Raw('/home/ninja/ninja.fif')
    data = raw.get_data()[raw.ch_names.index('EEG O1-Ref')]
    d = data[150*500: 190*500]
    tf = Morse(raw.info['sfreq'], cuda=True).power(d, np.arange(0.1, 50, 0.1))
    ax = plot_tf(tf, frange=(0, 50, 10), trange=(0, 40, 10), sfreq=500,
                 show=False)
    ax.set_title('My EEG power(O1)')
    ax.set_xlabel('Time Course(sec)')
    ax.set_ylabel('Hz')
    plt.show()


if __name__ == '__main__':
    # enable_cupy()
    print('Test Run')
    # plot_sin_fft()
    # test()
    if 'wave' in argv:
        simple_plot_test()
        test3d()
        fft_wavelet_test()
        other_wavelet_test()
    if 'cwt' in argv:
        cwt_test(False, True) if 'cuda' in argv else cwt_test(False, False)
        eeg()
