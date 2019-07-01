# NinWavelets
This is a python script to generate 'Generalized Morse Wavelets'.
It was written for mne python.
It is brand new project, and under heavily development.
Destructive changes may be made.

# Install
```
pip install git+https://github.com/uesseu/nin_wavelets
```

# Dependency
- Scipy
- numpy

Optional
- mne

```
pip install scipy
pip install numpy
pip install mne
```

# Exsample
It is similar to morlet wavelet, if you use default param.

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

# TODO

- Other wavelets
    + Morlet
    + Gabor
    + Mexican hat
    + Haar
    + etc...
- DWT
- Use cuda, cython and speedup!
- Kill typos(I am not good at English)

# Reference
## Morse Class
It is a class to generate morse wavelets.

```python
Morse(self, sfreq: float = 1000,
      b: float = 17.5, r: float = 3,
      length: float = 10, accuracy: float = 1) -> None:
```

Parameters

- sfreq: float | Sampling frequency. This behaves like sfreq of mne-python.
- b: float | beta value
- r: float | gamma value. 3 may be good value.
- accuracy: float | Accurancy paramater.
- length: float | Length paramater.

Accuracy and length is optional.
These are needed when you want to plot.

Beta and gamma can be set by anothor methods.
Beta and gamma is chainable.

```python
morse = Morse()
morse.beta(17.5).gamma(3)
```

## MorseMNE Class
It is same as Morse class.
But it uses mne.time_frequency.tfr.cwt to run cwt.
It is not recommended, because mne.time_frequency.tfr.cwt needs wavelet
which is not Fourier transformed.
Basically, GeneralizedMorseWavelets is a wavelet which is
'Fourier transformed wavelet' and so, you need to run
InverseFourier transform before you perform CWT.


### make_wavelets
```python
wavelet = Morse(1000, 17.5, 3).make_wavelets([10])[0]
```

Make morse wavelets.

Parameters
freq: float | Frequency. If frequency is too small,
It returnes bad wave easily.
For example, sfreq=1000, freq=3 it returnes bad wave.
If you want good wave, you must set large accuracy,
and length when you make this instance.

Returns
MorseWavelet: np.ndarray

### make_fft_waves
```
def make_fft_waves(self, total: float, one: float,
                   freqs: Iterable) -> Iterator:
```
Make Fourier transformed morse wavelet.

### cwt
#### Morse class
cwt method of Morse class.

```python
def cwt(self, wave: np.ndarray,
        freqs: Union[List[float], range, np.ndarray],
        max_freq: int = 0) -> np.ndarray:
```

example
```python
morse = Morse()
result = morse.cwt(sin, np.arange(1, 1000, 1))
plt.imshow(np.abs(result), cmap='RdBu_r')
plt.show()
```

max_freq is a param to cut result.

#### MorseMNE class
Same as mne.

```python
cwt(self, wave: np.ndarray,
    freqs: Union[List[float], range, np.ndarray], use_fft: bool = True,
    mode: str = 'same', decim: float = 1) -> np.ndarray:
```

### power
power(self, wave: np.ndarray,
      freqs: Union[List[float], range, np.ndarray]) -> np.ndarray:

Run cwt of mne-python, and compute power.

Parameters
freqs: float | Frequencies. Before use this, please run plot.

Returns
Result of cwt. np.ndarray.
