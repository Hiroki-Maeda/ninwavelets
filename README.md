# NinWavelets
This is a python script to generate 'Generalized Morse Wavelets'.
It was written for mne python.

# Install
```
pip install git+https://github.com/uesseu/nin_wavelets
```

# Dependency
Of cource, it depends on python, too.
mne python is needed.

```
pip install mne
```

# Exsample
It is similar to morlet wavelet, if you use default param.

```
morse = Morse(1000).beta(20)
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
    + Gover
    + Mexican hat
    + Haar
    + etc...
- CWT without mne
- Use cuda, cython and speedup!

# Reference
## Classes
### Morse
Morse(self, sfreq: float = 1000, b: float = 17.5, r: float = 3,
             length: float = 10, accuracy: float = 1) -> None:
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
    Morse instance its self.

## Morse method
### beta
beta(self, b: float) -> 'Morse':
Set gamma value of MorseWavelet.
If it is 17.5, MorseWavelets may resembles MorseWavelet with sigma 7.

Parameters
r: float | gamma value

Returns
Morse instance its self.

### gamma
gamma(self, r: float) -> 'Morse':

Set gamma value of MorseWavelet.
Good value may be 3.

Parameters
r: float | gamma value

Returns
Morse instance its self.

### wavelet
wavelet(self, freq: float = 10) -> np.ndarray:

Make morse wavelet.
It returnes one freq wavelet only, and so it may be useless.

Parameters
freq: float | Frequency. If frequency is too small,
    it returnes bad wave easily.
    For example, sfreq=1000, freq=3 it returnes bad wave.
    If you want good wave, you must set large accuracy, and length
    when you make this instance.

Returns
MorseWavelet: np.ndarray

### wavelets
wavelets(self, freqs: Union[List[float], range, np.ndarray]) -> np.ndarray:

Make morse wavelet.
It returnes list of wavelet, and it is compatible with mne-python.
(As argument of Ws of mne.time_frequency.tfr.cwt)

Parameters
freqs: List[float] | Frequency. If frequency is too small,
    it returnes bad wave easily.
    For example, sfreq=1000, freq=3 it returnes bad wave.
    If you want good wave, you must set large accuracy, and length
    when you make this instance.

Returns
MorseWavelet: np.ndarray

### cwt
cwt(self, wave: np.ndarray,
    freqs: Union[List[float], range, np.ndarray], use_fft: bool = True,
    mode: str = 'same', decim: float = 1) -> np.ndarray:

Run cwt of mne-python.

Parameters
freqs: float | Frequencies. Before use this, please run plot.

Returns
Result of cwt. Complex np.ndarray.

### power
power(self, wave: np.ndarray,
      freqs: Union[List[float], range, np.ndarray]) -> np.ndarray:

Run cwt of mne-python, and compute power.

Parameters
freqs: float | Frequencies. Before use this, please run plot.

Returns
Result of cwt. np.ndarray.
