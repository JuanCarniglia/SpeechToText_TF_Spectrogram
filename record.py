import pyaudio
import wave
import msvcrt
import uuid
import os


CHUNK = 1024 
FORMAT = pyaudio.paInt16 #paInt8
CHANNELS = 1 
RATE = 22050 #sample rate
WAVE_OUTPUT_FILENAME = "training" + os.sep + "training_sounds" + os.sep


print('Starting')

recording=False
newText = 'U'

while True:

    if recording:
        data = stream.read(CHUNK)
        frames.append(data) # 2 bytes(16 bits) per channel

    if msvcrt.kbhit():
        c = ord(msvcrt.getch())

        #print('Key : ' + str(c))
        if (c == 115):
            if recording:            
                stream.stop_stream()
                stream.close()
                p.terminate()

                class_dir = WAVE_OUTPUT_FILENAME + newText

                if not os.path.exists(class_dir):
                    os.mkdir(class_dir)

                fileName = uuid.uuid4().hex + '.wav'
                fileName = class_dir + os.sep + fileName

                wf = wave.open(fileName, 'wb')
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(p.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))
                wf.close()
                
                print("\tDone recording : " + newText)
                recording=False
            else:
                frames = []
                p = pyaudio.PyAudio()

                stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK) #buffer

                print("\tRecording : " + str(newText))

                recording=True
        elif (c == 113):
            break;
        elif (c == 110):
            #new file
            newText = input('New literal string : ')
            
    

