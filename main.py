
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

#! uncomment lines below if training for the first time (leave # # in place)

# # * looping through the text and
# for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE):
# 	sentences.append(text[i: i + SEQ_LENGTH])
# 	next_chars.append(text[i + SEQ_LENGTH])
 
# #* converting the lists into numpy arrays

# # create a 3D array of zeros with shape -> "number of sentences" x "sequence length" x "number of characters"
# x = np.zeros((len(sentences), SEQ_LENGTH, len(characters)), dtype=np.bool)

# # create a 2D array of zeros with shape -> "number of sentences" x "number of characters"
# y = np.zeros((len(sentences), len(characters)), dtype=np.bool)

# for i, sentence in enumerate(sentences):
#     for t, char in enumerate(sentence):
#         x[i, t, char_2_index[char]] = 1
#     y[i, char_2_index[next_chars[i]]] = 1

# #> --- --- --- Creating the model for the first time --- --- --- <#

# model = Sequential()
# model.add(LSTM(128, input_shape=(SEQ_LENGTH, len(characters))))
# model.add(Dense(len(characters)))
# model.add(Activation('softmax'))

# #* Compiling the model
# #	loss: loss function
# #	optimizer: optimizer
# #	metrics: list of metrics to be evaluated and displayed

# model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.01))
# #xx model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.01), metrics=['accuracy'])

# #* fitting the model

# Set the batch size for training
# A larger batch size can speed up training, but may also require more memory
# A smaller batch size can be more memory-efficient, but may also require more iterations to converge on a solution
# In general, a larger batch size is recommended for larger datasets, while a smaller batch size is recommended for smaller datasets or when memory is limited
batch_size = 128 #? try different batch sizes like 32, 64, 128, 256, 512

# Set the number of epochs for training
# A larger number of epochs can allow the model to learn more complex patterns in the data, but may also increase the risk of overfitting
# In general, a larger number of epochs is recommended for larger datasets, while a smaller number of epochs is recommended for smaller datasets or when overfitting is a concern
epochs = 3 #? try different number of epochs like  1, 3, 4, 6, 7, 9

#* Train the model with specified batch size and number of epochs

# model.fit(x, y, batch_size=batch_size, epochs=epochs)
# #xx model.fit(x, y, batch_size=256, epochs=4)

# model.save('text_gen.model')

#> --- --- --- Loading the model --- --- --- <#

model = tf.keras.models.load_model('text_gen.model')

def sample(preds, temperature = 1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

#> --- --- --- Generating text --- --- --- <#

def generate_text(length, temperature):
    start_index = rd.randint(0, len(text) - SEQ_LENGTH - 1)
    generated = ''
    sentence = text[start_index: start_index + SEQ_LENGTH]
    generated += sentence
    for i in range(length):
        x_pred = np.zeros((1, SEQ_LENGTH, len(characters)))
        for t, char in enumerate(sentence):
            x_pred[0, t, char_2_index[char]] = 1.

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, temperature)
        next_char = index_2_char[next_index]
        
        generated += next_char
        sentence = sentence[1:] + next_char

    return generated

#> --- --- --- Generating text --- --- --- <#

print("--- --- --- Generating text, temp = 0.2 --- --- ---\n")
print(generate_text(500, 0.2))

print("--- --- --- Generating text, temp = 0.5 --- --- ---\n")
print(generate_text(500, 0.5))

print("--- --- --- Generating text, temp = 0.6 --- --- ---\n")
print(generate_text(500, 0.6))

print("--- --- --- Generating text, temp = 0.9 --- --- ---\n")
print  (generate_text(500, 0.9))

print("--- --- --- Generating text, temp = 1.0 --- --- ---\n")
print(generate_text(500, 1.0))