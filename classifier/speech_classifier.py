# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import ml_helper
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='9'
import tflearn
from tflearn.data_utils import shuffle

from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization

from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import numpy as np
from pathlib import Path
import argparse
import time
import json
from watchdog.observers import Observer  
from watchdog.events import PatternMatchingEventHandler

app_path = os.getcwd()

dimsX = 269
dimsY = 199

def predict(image_file):
    from PIL import Image, ImageOps
    global dimsX
    global dimsY
    global arr_labels
    global verbose
    
    start_time = time.time()

    #print('Image: ' + image_file)
    im = Image.open(image_file).convert('RGB')

    print('Start processing File: ' + image_file.replace(app_path, ''))
            
    json_str = '['

    imgwidth, imgheight = im.size

    print(im.size)
    
    tiles = 0
    sum = 0

    # Esto lo podria cambiar a
    # img_arr = img_arr.reshape(imgwidth/64, 64,64,3).astype(np.float)
    # y no hacer el for ?

    # probar predict_label(img_arr)
    
    _hash = ml_helper.find_between(image_file, 'IMAGES_TO_CHECK' + os.sep, '.')

    for i in range(0, imgwidth, dimsX):
        box = (i, 0, i+dimsX, dimsY)
        a = im.crop(box)

        Img_capped = ImageOps.fit(a, ((dimsX,dimsY)), Image.ANTIALIAS)
        img_arr = np.array(Img_capped)
        img_arr = img_arr.reshape(1,dimsX,dimsY,3).astype(np.float)

        prediction = model.predict(img_arr)

        print(prediction)
        
        pred = np.argmax(prediction[0])

        if (pred == 0 and prediction[0][pred] == 0): # or suma == count:
            pred = -1

        if not json_str == '[':
            json_str += ', '

        str_pred = "[["
        for x in prediction[0]:
            if not str_pred == "[[":
                str_pred += ", "
            str_pred += str(format(x, 'f'))

        str_pred += "]]"
        
        json_str += '{ "img" : "' + str(int(i/dimsX)) + '", "prediction_array" : "' + str_pred + '", "result" : "'

        json_str += str(pred) + '"}'

        if pred > -1:
            class_dir = app_path + os.sep + 'classification_results' + os.sep + 'IMAGES_TO_CHECK' + os.sep + arr_labels[pred]
        else:
            class_dir = app_path + os.sep + 'classification_results' + os.sep + 'IMAGES_TO_CHECK' + os.sep + 'undetected'
        
        
        if not os.path.exists(class_dir):
            os.mkdir(class_dir)
            
        a.save(class_dir + os.sep + _hash + '_IMG_X'+str(i)+'.png')

        tiles+=1

        
    json_str += ']'

    text_file = open(app_path + os.sep + 'classification_results' + os.sep + 'IMAGES_TO_CHECK' + os.sep + _hash + '_IMG' + str(tiles) + '.json', "w")
    text_file.write(json_str)
    text_file.close()

    elapsed_time = time.time() - start_time
    print('Generated File: ' + _hash + '_IMG' + str(tiles) + '.json' + '. Total : ' + str(tiles) + ' tiles in ' + str(round(elapsed_time,4)) + ' secs.\r\n')

    if verbose:
        print('JSON output')
        print(json.dumps(json_str, indent=4, sort_keys=True))


class MyHandler(PatternMatchingEventHandler):
    patterns = ["*.jpg"]

    def process(self, event):
        global obs_folder
        """
        event.event_type 
            'modified' | 'created' | 'moved' | 'deleted'
        event.is_directory
            True | False
        event.src_path
            path/to/observed/file
        """
        
        if event.src_path.find('_IMG_X') == -1 and event.src_path.find('_MARKS') == -1:
            time.sleep(2)
            print('Processing file : ' + str(event.src_path))
            predict(event.src_path)
        
    def on_created(self, event):
        self.process(event)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--img", help = "Path to the image file with all the tiles.")
#ap.add_argument("-t", "--tfl", help = "Path to the TFL File.")
ap.add_argument("-v", "--verbose", help="Verbose output")
args = vars(ap.parse_args())


verbose = args.get("verbose", False)
#tfl_path = args.get("mask", False)
image = args["img"]

if verbose:
    print('Verbose Output. JSON Will be printed out')

classes = 0

arr_labels = []
for subdir, dirs, files in os.walk('..' + os.sep + 'training' + os.sep + 'training_images_plot'):
    classes = len(dirs)
    arr_labels = dirs
    break

# Same network definition as before
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

img_aug = ImageAugmentation()
#img_aug.add_random_flip_leftright()
#img_aug.add_random_flip_updown()
#img_aug.add_random_rotation(max_angle=25.)
img_aug.add_random_blur(sigma_max=3.)

network = input_data(shape=[None, dimsX, dimsY, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)

network = conv_2d(network, 128, 11, strides=4, activation='relu')
network = max_pool_2d(network, 3)
network = local_response_normalization(network)
network = conv_2d(network, 256, 5, activation='relu')
network = max_pool_2d(network, 3)
network = local_response_normalization(network)
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 3)
network = local_response_normalization(network)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, classes, activation='softmax')
network = regression(network, optimizer='momentum',
                     loss='categorical_crossentropy',
                     learning_rate=0.00001)

model = tflearn.DNN(network, tensorboard_verbose=2, max_checkpoints=1,
                    checkpoint_path='..' + os.sep + 'training' + os.sep + 'speech-classifier.tfl.ckpt',
                    tensorboard_dir='tmp/tflearn_logs/')

model.load('..' + os.sep + 'training' + os.sep + 'speech-classifier.tfl')

print('Model loaded. Ready to process\r\n')
obs_folder = app_path + os.sep + 'classification_results' + os.sep + 'IMAGES_TO_CHECK'
print('Observing folder : ' + obs_folder)

observer = Observer()
observer.schedule(MyHandler(), path=obs_folder, recursive=False)
observer.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    observer.stop()

observer.join()
