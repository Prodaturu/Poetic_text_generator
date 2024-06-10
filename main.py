
#> --- --- --- Importing Libraries --- --- --- <#

import random as rd
import tensorflow as tf
from tensorflow.keras.models import sequential
from tensorflow.keras.layers import Dense, LSTM, Activation
from tensorflow.keras.optimizers import RMSprop
import numpy as np

#> --- --- --- Importing data --- --- --- <#

filepath = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')  # Loads data from url

#* read the data in file
#    .decode is used to decode the file using a specific encoding

text = open(filepath, 'rb').read().decode(encoding='utf-8')

text = text[400000:700000] # character from where to where we want to train our model on

# creates a set out of the text using `set(text)` -> converts it into a list using `list(..)`  -> sorts it with `sorted(...)`
characters = sorted(list(set(text))) # list of all characters in the text

char_2_index = dict((c, i) for i, c in enumerate(characters)) # maps each character to an index
index_2_char = dict((i, c) for i, c in enumerate(characters)) # maps each index to a character

#> --- --- --- Preparing data --- --- --- <#

SEQ_LENGTH = 20
STEP_SIZE = 1

#* creating empty list of sentences and targets

sentences = []
next_chars = []

# * looping through the text and
for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE):
	sentences.append(text[i: i + SEQ_LENGTH])
	next_chars.append(text[i + SEQ_LENGTH])
 
#* converting the lists into numpy arrays

# create a 3D array of zeros with shape -> "number of sentences" x "sequence length" x "number of characters"
x = np.zeros((len(sentences), SEQ_LENGTH, len(characters)), dtype=bool)

# create a 2D array of zeros with shape -> "number of sentences" x "number of characters"
y = np.zeros((len(sentences), len(characters)), dtype=bool)