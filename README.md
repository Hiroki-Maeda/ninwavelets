# NinWavelets
This is a python package to generate 'Generalized Morse Wavelets'(GMW).
And perform CWT based on GMW.

# Install
```
pip install git+https://github.com/uesseu/nin_wavelets
```

# Dependency
- python 3.6.5 or newer

These are automatically installed.
- Scipy
- numpy
- cython

Optionally, you can use this.
- mne

```
pip install mne
```

At first, it was written for mne python, but using mne function is ugly way.  
(Because it needs inverse Fourier transform to no purpose.)  
Now it has own CWT method.  
It is brand new project, and under heavily development.  
Destructive changes may be made.  


# Exsample
GMW is similar to morlet wavelet, if you use default param.

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


# Reference
## Morse Class
This is a class to GMW.

```python
from nin_wavelets import Morse
```

```python
Morse(self, sfreq: float = 1000,
      b: float = 17.5, r: float = 3,
      length: float = 10, accuracy: float = 1) -> None:
```

Parameters

| Param    | Type  | Default |                                                              |
| --       | --    | --      | --                                                           |
| sfreq    | float | 1000Hz  | Sampling frequency.                                          |
| b        | float | 17.5    | beta value                                                   |
| r        | float | 3       | gamma value. 3 may be good value.                            |
| accuracy | float | 1       | Accurancy paramater. It affects only when you plot wavelets. |
| length   | float | 10      | Length paramater. It affects only when you plot wavelets.    |



```python
morse = Morse()

```


### make_wavelets
```python
wavelet = Morse(1000, 17.5, 3).make_wavelets([10])[0]
```

Make list of morse wavelets.  

| Param | Type  |                      |
|-------|-------|----------------------|
| freq  | float | List of frequencies. |

Because it returnes bad wave easily,  
you should use it when you plot only.  
For example, sfreq=1000, freq=3 it returnes bad wave.  
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

| Param              | Type  |                                               |
|--------------------|-------|-----------------------------------------------|
| wave               | float | Wave drawed by numpy.                         |
| freqs              | float | List of frequencies.                          |
| max_freq           | float | Max freq.                                     |
| kill_nikist(False) | bool  | Whether kill nikist freqs of wave to analyze. |

```python
def cwt(self, wave: np.ndarray,
        freqs: Union[List[float], range, np.ndarray],
        max_freq: int = 0) -> np.ndarray:
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
      freqs: Union[List[float], range, np.ndarray]) -> np.ndarray:
```

Run cwt of mne-python, and compute power.

| Param              | Type  |                                               |
|--------------------|-------|-----------------------------------------------|
| wave               | float | Wave drawed by numpy.                         |
| freqs              | float | List of frequencies.                          |
| max_freq           | float | Max freq.                                     |
| kill_nikist(False) | bool  | Whether kill nikist freqs of wave to analyze. |

Returns  
Result of cwt. np.ndarray.  

## MorseMNE Class
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

By the way, Morlet Wavelet has formula which is Fourier transformed.  
And so, I think, it may be better to use the formula  
even if you use Morlet Wavelet.  

# Scalability
This package is scalable.
You can inherit nin_wavelets.BaseWavelet class.  
Then, you can edit these methods.  

- BaseWavelet.wavelet_formula
- BaseWavelet.trans_wavelet_formula
- BaseWavelet.peak_freq

At first, you need to overwrite them.
They needs to written by numpy.
These methods are used in the class, and bothering procedures are done.


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
    + [ ] Do you need 3D wavelet?
- [x] Use cuda, cython and speedup!
    + [x] It was cythonized and may be faster. I think ;)
    + [ ] I already tried cuda and it was slow...;(
- [ ] Kill typos(I am a bad male yellow monkey and not good at English) ;(
- [ ] Licence
    + [ ] Whether write my name or not.
    + [ ] Which licence to use(I like MIT)
