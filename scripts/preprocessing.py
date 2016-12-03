# This script will contain the functions and algorithms for preprocessing the raw audio data into training and testing datasets

# path to folder of sounds
	# get # of sounds
	# iterate over sounds
	# get sample audio data
	# pad sample start with zeros (L = number of samples being fed to base layer)

import os
import wave

def getData():
	audioData = []
	path = '/Users/alex/projects/nn-sampler/scripts/kicks/'
	#audioData = wave.open(path)
	startFolder = os.listdir(path)
	for file in startFolder:
		#print(file)
		if (file.endswith('.wav')):
			wav = wave.open(path + file)
			audioData.append(wav.readframes(wav.getnframes()))

	#print(len(audioData))

	#print(audioData[0][0])
	#function, grabsound(), that gets a sound from the folder
	#applies 'mu-law companding transformation' 
	#quantizes to 256 values

	#function createData()
	return audioData

x = getData()
print(x[0][0])