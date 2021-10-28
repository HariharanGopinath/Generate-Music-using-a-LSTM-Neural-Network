# Generate-Music-using-a-LSTM-Neural-Network
Nowadays, deep machine learning seems to be playing a significant role in many aspects of almost every field like image detectors, computer vision, medical and business analytic. But a multitude of the available models has been designed for image processing/ classification or text-processing where mostly a variant of CNN or feed-forward neural network is used to extract more in-depth features. However, there are many other realistic problems, where further research need to be carried out on deep machine learning models to address some prominent issues. One of these domains (which require further consideration) is that of music generation. Traditionally, music was treated as an analogue signal and was generated manually. However, in recent years, music is conspicuous to technology that can generate a suite of music automatically without any human intervention. To accomplish this task, we need to overcome some technical challenges to successfully generate a composition of notes having continuity and unity. This is where we can try for different Deep learning frameworks on sequential inputs other than images.

# Dataset 
For the dataset we have used used piano music from
the soundtrack of Final fantasy as it seems to have distinct
melodies that is present in majority of the music pieces. We
are using MIDI dataset here as the other datasets of music
like the mp3, au etc are raw audio files with a lot of noise
which messes up the pre process. For examining the data we
are going to be using here, Music21 a python toolkit which is
used to get the musical notations of a MIDI file. To make our
own midi files in the final music creation process it helps us
by creating the Note and Chord objects from the dataset here.
The Note object consists of : 1)Pitch - frequency of the sound
represented with the letters [A, B, C, D, E, F, G] 2)Octave -
the set of pitches which is used on the piano here 3)Offset - the
location of the note in the piece. The Chord objects behaves
as a container for the set of notes that is being played at a
given time.

![Project_poster group 110-1](https://user-images.githubusercontent.com/71879067/139288699-51e0a4e1-f610-4983-8a3b-ff56d1569f00.jpg)


# MODEL
The picture of the LSTM can be seen below



# RESULTS
<img width="501" alt="accuracy_100epo_midisongs" src="https://user-images.githubusercontent.com/71879067/139288982-d970ef22-ee4a-46cc-9c88-25a7f0a83fec.png">
<img width="494" alt="loss_100epo_midisongs" src="https://user-images.githubusercontent.com/71879067/139289013-70018be2-4cd0-4fde-9240-6859f6381034.png">

