import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np
import os

def graph_spectrogram(wav_file):
    rate, data = get_wav_info('samples' + os.sep + 'english_male' + os.sep + wav_file + '.wav')

    plt.axis('off')

    dt = 0.0005

    NFFT = 512       # the length of the windowing segments
    Fs = int(1.0/dt)  # the sampling frequency

    Pxx, freqs, bins, im = plt.specgram(data, NFFT=NFFT, Fs=Fs, noverlap=511)

    #plt.show()
    
    plt.savefig(wav_file + '.jpg',
                dpi=50, # Dots per inch
                frameon='false',
                aspect='normal',
                bbox_inches='tight',
                pad_inches=0) # Spectrogram saved as a .png 

def get_wav_info(wav_file):
    rate, data = wavfile.read(wav_file)
    return rate, data

if __name__ == '__main__': # Main function
    wav_file = 'day' # Filename of the wav file
    graph_spectrogram(wav_file)
