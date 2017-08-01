import matplotlib.pyplot as plt
from scipy.io import wavfile
import wave
import numpy as np
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='9'

def graph_spectrogram(wav_file):
    rate, data = get_wav_info(wav_file + '.wav')
    nfft = 256  # Length of the windowing segments
    fs = 64   # Sampling frequency
    pxx, freqs, bins, im = plt.specgram(data, nfft,fs)
    plt.axis('off')
    plt.savefig(wav_file + '_spec.jpg',
                dpi=300, # Dots per inch
                frameon='false',
                aspect='normal',
                bbox_inches='tight',
                pad_inches=0) # Spectrogram saved 

def graph_soundwave(wav_file):
        
    spf = wave.open(wav_file + '.wav','r')

    #Extract Raw Audio from Wav File
    signal = spf.readframes(-1)
    signal = np.fromstring(signal, 'Int16')


    #If Stereo
    if spf.getnchannels() == 2:
        print('Just mono files')
        sys.exit(0)

    #plt.figure(1)
    #plt.title('Signal Wave...')
    plt.plot(signal)
    plt.axis('off')
    #plt.show()
    plt.savefig(wav_file + '_plot.jpg',
                dpi=300, # Dots per inch
                frameon='false',
                aspect='normal',
                bbox_inches='tight',
                pad_inches=0) # Spectrogram saved as a .png 

def get_wav_info(wav_file):
    rate, data = wavfile.read(wav_file)
    return rate, data

if __name__ == '__main__': # Main function
    wav_file = 'training/training_sounds/casa/0ecc3c3906f84372a876c4a86bdeda2d' # Filename of the wav file
    graph_spectrogram(wav_file)
    graph_soundwave(wav_file)
