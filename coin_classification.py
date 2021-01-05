#import libraries 
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from os import listdir
from os.path import isfile, join
#import data 
#create a list of filenames of images and a list of labels

authorities_vals = {
    'Alexander' : 0,
    'Ptolemy' : 1,
    'Antiochus' : 2,
    'Seleucus' : 3
}

filenames = []
labels = []

def GetData(authority_name):
    for file in os.listdir('images/' + authority_name):
        filenames.append('images/' + authority_name + file)
        labels.append(authorities_vals[authority_name])

GetData('Alexander')
GetData('Ptolemy')
GetData('Antiochus')
GetData('Seleucus')
    
#create a dataset returning slices of 'filenames'
dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

#parse every image in the dataset using 'map'
def ParseFunction(filenames, labels):
    image_string = tf.read_file(filenames)
    image_decoded = tf.image.decode_jpeg(image_string, channels = 4)
    image = tf.cast(image_decoded, tf.float32)
    return image, labels
ParseFunction(filenames, labels)
print(image)
dataset = dataset.map(ParseFunction)
dataset = dataset.batch(4)

#create iterator and final input tensor
iterator = dataset.make_one_shot_iterator()
image, labels = iterator.get_next()






