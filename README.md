# NinWavelets
This is a python package for wavelet transform.
Generalized Morse(GMW), Morlet and so on.
It can also perform CWT based on GMW.

# Why NinWavelets?
- Use wavelets which is originally Frourier transformed
    + Generalized Morse Wavelets, and so on.
- Skipping one FFT when performing CWT.
- Scalable(?)


# Install
```
pip install git+https://github.com/uesseu/nin_wavelets
```

# Dependency
- python 3.6.5 or newer

These are automatically installed.
- Scipy
- numpy

Optionally, you can use this.
- mne

```
pip install mne
```

At first, it was written for mne python.  
but using mne function with this package is ugly way.  
(Because it needs inverse Fourier transform to no purpose.)  
Now it has own CWT method.  
It is brand new project, and under heavily development.  
Destructive changes may be made.  


# Exsample
GMW is similar to morlet wavelet, if you use default param.

You can calculate power.
```
morse = Morse(1000, gamma=3, beta=17.5)
freq = 60
time = np.arange(0, 0.3, 0.001)
sin = np.array([np.sin(time * freq * 2 * np.pi)])

result = morse.power(sin, range(1, 100))
plt.imshow(result, cmap='RdBu_r')
plt.gca().invert_yaxis()
plt.title('CWT of 60Hz sin wave')
plt.show()
```

You can also calculate inter trial coherence.

```python
result = morse.power(sin, range(1, 100))
```

You can also use plot_tf().
It can plot result with colorbar.

```python
from nin_wavelets import plot_tf
plot_tf(result)
```

If you just want to perform cwt only, write like this.

```python
result = morse.cwt(sin, range(1, 100))
```

If you are mne user, epochs can be processed.
See 'NinWavelets for MNE'.


# Reference
## WaveletBase Class
Super class of wavelets.
You can inherit this class and make new wavelets.

After inherit this, you can edit these methods.  

- BaseWavelet.wavelet_formula
- BaseWavelet.trans_wavelet_formula
- BaseWavelet.peak_freq

At first, you need to overwrite them.
They needs to written by numpy.
These methods are used in the class,
and bothering procedures are done.

## Way to inherit

This is an example of __init__.

```python
def __init__(self, sfreq: float = 1000, b: float = 17.5, r: float = 3,
             accuracy: float = 1, real_wave_length: float = 1.,
             interpolate: bool = False) -> None:
    super(Morse, self).__init__(sfreq, accuracy,
                                real_wave_length, interpolate)
    self.r: float = r
    self.b: float = b
    self.mode = WaveletMode.Reverse
```



## Morse Class

This is a class to GMW.
Sub class of WaveletBase.

```python
from nin_wavelets import Morse
```

```python
Morse(self, sfreq: float = 1000,
      b: float = 17.5, r: float = 3,
      length: float = 10, accuracy: float = 1,
      interpolate=False) -> None:
```

Parameters

| Param       | Type  | Default |                                                              |
| --          | --    | --      | --                                                           |
| sfreq       | float | 1000Hz  | Sampling frequency.                                          |
| b           | float | 17.5    | beta value                                                   |
| r           | float | 3       | gamma value. 3 may be good value.                            |
| accuracy    | float | 1       | Accurancy paramater. It affects only when you plot wavelets. |
| length      | float | 10      | Length paramater. It affects only when you plot wavelets.    |
| interpolate | bool  | False   | Interpolate frequencies which is higher than nyquist freq.   |



```python
morse = Morse()
```

interpolate=True makes your code faster.
(Up to half time)
But, I dont know whether it is good or bad...

### make_wavelets

Exsample.

```python
wavelet = Morse(1000, 17.5, 3).make_wavelets([10])[0]
```

Make list of wavelets.  

| Param | Type  |                      |
|-------|-------|----------------------|
| freq  | float | List of frequencies. |

Because it returnes bad wave easily,  
you should use it when you plot it only.  
For example, GMW with sfreq=1000, freq=3 returnes bad wave.  
If you want good wave, you must set  
large accuracy and length when you make this instance.  

Returns  
MorseWavelet: list of np.ndarray  

### make_fft_waves

```python
make_fft_waves(self, total: float, one: float,
               freqs: Iterable) -> Iterator:
```
Make Fourier transformed Wavelet.
If the wavelet is originally Frourier transformed wavelet,
it just calculate original formula.
If wavelet is originally not Fourier transformed wavelet,
it run FFT to make them.

### cwt
CWT method.

| Param    | Type  |                                                                      |
|----------|-------|----------------------------------------------------------------------|
| wave     | float | Wave drawed by numpy.                                                |
| freqs    | float | List of frequencies.                                                 |
| max_freq | float | Max freq.                                                            |
| reuse    | bool  | Reuse wavelets you made before. If true, calculation becomes faster. |

```python
def cwt(self, wave: np.ndarray,
        freqs: Union[List[float], range, np.ndarray],
        max_freq: int = 0, reuse=True) -> np.ndarray:
```

example
```python
import numpy as np
freq: float = 60
length: float = 5

time: np.ndarray = np.arange(0, length, 0.001)
sin = np.array(np.sin(time * freq * 2 * np.pi))
morse = Morse()
result = morse.cwt(sin, np.arange(1, 1000, 1))
plt.imshow(np.abs(result), cmap='RdBu_r')
plt.show()
```

max_freq is a param to cut result.

## power
```
power(self, wave: np.ndarray,
      freqs: Union[List[float], range, np.ndarray],
      reuse=True) -> np.ndarray:
```

Run cwt of mne-python, and compute power.

| Param    | Type  |                                                                      |
|----------|-------|----------------------------------------------------------------------|
| wave     | float | Wave drawed by numpy.                                                |
| freqs    | float | List of frequencies.                                                 |
| max_freq | float | Max freq.                                                            |
| reuse    | bool  | Reuse wavelets you made before. If true, calculation becomes faster. |

Returns  
Result of cwt. np.ndarray.  

## MorseMNE Class(Bad way)
MorseMNE class to use function of MNE-python,  
which is Great package to analyze EEG/MEG.  
It is same as Morse class except cwt but  
if you run cwt, it uses mne.time_frequency.tfr.cwt to run cwt.  

But it is not recommended, because mne.time_frequency.tfr.cwt needs  
wavelet which is 'not Fourier transformed'.  
Basically, GMW is a wavelet which is originally  
'Fourier transformed wavelet' and so, you need to run  
InverseFourier transform before you perform CWT.  
I think, this ugly class is disgusting.  

By the way, there is a formula of Morlet wavelet which is Fourier transformed.  
And so, I think, it may be better to use the formula  
even if you use Morlet Wavelet.  

## NinWavelets for MNE
nin_wavelets.EpochsWavelet is a class for Epochs class of mne.

'''python
from nin_wavelets import EpochsWavelet, Morse, plot_tf
from mne import read_epochs

fname = 'hoge_epo.fif'
epochs = read_epochs(fname)
morse = Morse()
result = EpochsWavelet(epochs, morse).power(range(1, 100))
plot_tf(result)
'''

At first, make instance of wavelets(Morse, Morlet and so on).
Then, make EpochsWavelet class.
This has methods named cwt, power and itc.
plot_tf is a function to plot numpy array.


# Licence
'This software is released under the MIT License, see LICENSE.txt.'  
I thought so. But, tellilng you my name needs courage.  
I am thinking about it...

# TODO

- Other wavelets
    + [x] Morse
    + [x] Morlet
    + [x] Gabor
    + [ ] Mexican hat
    + [ ] Haar
    + [ ] Scalability for unknown wavelets
- More methods
    + [ ] Decimation
    + [ ] DWT
    + [ ] 2D wavelet
- [ ] Use cuda, cython and speedup!
    + [ ] It was cythonized before, but not very fast. Now, it is pure python.
    + [ ] I already tried cuda and it was extremely slow...;(
- [ ] Kill typos(I am a bad male yellow monkey and not good at English) ;(
- [ ] Licence
    + [ ] Whether write my name or not.
    + [ ] Which licence to use.
