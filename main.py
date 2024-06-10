# --- --- --- Importing Libraries --- --- --- #

import random as rd
import tensorflow as tf
from tensorflow.keras.models import sequential
from tensorflow.keras.layers import Dense, LSTM, Activation
from tensorflow.keras.optimizers import RMSprop

# --- --- --- Importing data --- --- --- #

filepath = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')  # Loads data from url

# read the data in file
#    .decode is used to decode the file using a specific encoding
text = open(filepath, 'rb').read().decode(encoding='utf-8')

text = text[400000:700000] # character from where to where we want to train our model on

# creates a set out of the text using `set(text)` -> converts it into a list using `list(..)`  -> sorts it with `sorted(...)`
characters = sorted(list(set(text))) # list of all characters in the text

char_2_index = dict((c, i) for i, c in enumerate(characters)) # maps each character to an index