
import numpy
import glob
import pickle
import pickle
from music21 import instrument, note, stream, chord
import numpy
from music21 import converter, instrument, note, chord

from keras.layers import BatchNormalization as BatchNorm
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt

import numpy as np
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense


def generate():
    """ Generate a piano midi file """
    #load the notes used to train the model
    with open('data/midi_songs_new50', 'rb') as filepath:
        notes = pickle.load(filepath)

    # Get all pitch names
    pitchnames = sorted(set(item for item in notes))
    # Get all pitch names
    n_vocab = len(set(notes))

    network_input, normalized_input = input_output_sequences(notes)
    model = create_network(normalized_input, n_vocab)
    prediction_output = generate_notes(model, network_input, pitchnames, n_vocab)
    create_midi(prediction_output)

def input_output_sequences(notes):
   
    #the sequence length is used in the sequence the network which mainly to predict the next notes with the help of the previous 100 notes that would help to make the prediction of music sequence.
    length_seq = 100
    
    # creating a empty sequency for the both input and the output
    input_sequence = []
    output_coded = []

    # we storing the collection of the  multiple notes and chords in the notes variable like
    # we will have many repeated names of notes and chords that are occuring in the music and we are storing that here (ie) pitches in the music

    pitch = sorted(set(item for item in notes))
    n=len(pitch)

    # creating  a dict which will map or join the pitches and its intgers
    #note_to_int = dict((note, number) for number, note in enumerate(pitch))
    a=[]
    b=[]
    for inte, note in enumerate(pitch):
      a.append(inte)
      b.append(note)
    a_dict = {value: key for key, value in zip(a, b)}
      
    #print(a_dict)
    input_lenght=len(notes)-100

    # create input sequences and the corresponding outputs
    input_sequence=[]
    for i in range(0, input_lenght):
        #print(i)
        # for input sequence
        input = notes[i:i + length_seq]
        input_sequence.append([a_dict[char] for char in input])
        #for char in input:
         # input_sequence.append([a_dict[char]])
        # for output sequence
        out = notes[i + length_seq]
        output_coded.append(a_dict[out])


    # reshape the input for the lstm sequence
    normalized = numpy.reshape(input_sequence, (len(input_sequence), length_seq, 1))
    # normalize input

    normalized = normalized / float(n)
    #one-hot encode the output.

    output_coded = np_utils.to_categorical(output_coded)

    return (input_sequence, normalized)

def create_network(network_input, n_vocab):
    """ create the structure of the neural network """
    from keras.layers import Activation, Dense, LSTM, Dropout, Flatten
    model = Sequential()
    #layer LSTM one
    model.add(LSTM(128, input_shape=network_input.shape[1:], return_sequences=True))
    # dropout layer
    model.add(Dropout(0.2))
     #layer LSTM 2
    model.add(LSTM(128, return_sequences=True))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Load the weights to each node
    model.load_weights('weights_midi_songs_new50.hdf5')

    return model

#---------------------------------

def generate_notes(model, network_input):
  # Pick a random integer as the start of the prediction
  start = np.random.randint(0, len(network_input)-1)  
  pitch = sorted(set(item for item in notes))
  n=len(pitch)

  # creating  a dict which will map or join the integers and its notes
  a=[]
  b=[]
  for inte, note in enumerate(pitch):
    a.append(inte)
    b.append(note)
  a_dict = {value: key for key, value in zip(a, b)}
      

  input_lenght=len(notes)-100

  pattern = network_input[start]  
  pattern_len=len(pattern)
  output_pred = []  
  
  # generate 500 notes as it gives enough time for a melody to be created

  for note_index in range(500):
    #for input prediction and reshaping the pattern for the input prediction
    input_pred = np.reshape(pattern, (1, pettern_len, 1))  
    input_pred = input_pred / float(n)
      
    #numpy array of predictions
    index = numpy.argmax(model.predict(input_pred, verbose=0))
    #the note with the highest probability is indexed and given as the output
    output_pred.append(a_dict[index])


    pattern.append(index)
    pattern = pattern[1:len(pattern)]

  return output_pred



def create_midi(output_pred):
    #midi file created from the notes which are the output from the prediction and converted
    offset = 0
    notes_output = []

    # values generated by the model is used to create note object and chord object
    for pattern in output_pred:
        # pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            #array of notes created by the splitting of the string
            chord_notes = pattern.split('.')
            notes = []
            #loop for the string representation of every note
            for current_note in chord_notes:
                #note object created for every string representation of every note
                note_new = note.Note(int(current_note))
                note_new.storedInstrument = instrument.Piano()
                notes.append(new_note)
            #chord object created to contain each of these notes
            chord_new = chord.Chord(notes)
            chord_new.offset = offset
            notes_output.append(chord_new)
        # Note object created by the string representation of pitch in the pattern
        else:
            note_new = note.Note(pattern)
            note_new.offset = offset
            note_new.storedInstrument = instrument.Piano()
            notes_output.append(new_note)

        # end of every iteration by 0.5 and the note and chord object is appended to a list
        offset += 0.5
    #the output list is not used to create Music21 stream
    stream_midi = stream.Stream(notes_output)
    #the music generated by the network is stored in a midi file
    stream_midi.write('midi', fp='songs_piano_40epochs.mid')
    



#---------------------------------


if __name__ == '__main__':
    generate()
