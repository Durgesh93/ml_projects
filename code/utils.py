import os
import librosa
import librosa.display
import numpy as np



"""
"""
def spectrogram(data,n_fft= 2048,fftSize= 2048,overlap= -1,window='hann',center=True):

    if window == 'hann':
        win = librosa.filters.get_window('hann',fftSize)
    else:
        raise NotImplementedError
    
    if overlap == -1:
        jump = fftSize//4
    else:
        jump = int(fftSize*(100-overlap)/overlap)
    D = librosa.stft(data,center=center,window=win,win_length=fftSize,n_fft=n_fft,hop_length=jump)
    return D
"""
"""
def mel_spectrogram(D,pwr=2,sr=44100):
    if pwr == 2:
        D = np.abs(D)**2
    elif pwr == 1:
        D = np.abs(D)**1
    S = librosa.feature.melspectrogram(S=D,power=pwr,sr=sr)
    return S 

"""
"""
def reduce_bg_noise(y):
    import noisereduce as nr
    return nr.reduce_noise(audio_clip=y, noise_clip=y, verbose=False)

"""
"""
def amp(K):
    return librosa.amplitude_to_db(np.abs(K),ref=np.max(K))




