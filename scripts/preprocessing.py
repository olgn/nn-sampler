# This script will contain the functions and algorithms for preprocessing the raw audio data into training and testing datasets

# path to folder of sounds
	# get # of sounds
	# iterate over sounds
	# get sample audio data
	# pad sample start with zeros (L = number of samples being fed to base layer)

import os
import wave
import numpy as np

windowSize = 512	

def getData():
	audioFiles = []
	path = '/Users/alex/projects/nn-sampler/scripts/kicks/'
	#audioData = wave.open(path)
	startFolder = os.listdir(path)
	for file in startFolder:
		#print(file)
		if (file.endswith('.wav')):
			wav = wave.open(path + file)
			audio = wav.readframes(wav.getnframes())
			print(audio[0])
			audioFiles.append(audio)

	numFiles = len(audioFiles)

	Y = [] 
	X = []

	for file in range(numFiles):
		audioData = audioFiles[file]
		totalSamples = len(audioData)
		inputWindow = []

		for t in range(totalSamples - 1):
			left = t - windowSize - 1
			right = t

			if (np.mod(right, 100) == 0):
				print('processed' , right, 'rows of data')

			for n in range(0,windowSize):
				idx = left + n
				if idx >= 0:
					inputWindow.append(audioData[idx])
				else:
					inputWindow.append(0)

			X.append(inputWindow)
			output = audioData[right]
			Y.append(output)
	#print(len(audioData))

	#print(audioData[0][0])
	#function, grabsound(), that gets a sound from the folder
	#applies 'mu-law companding transformation' 
	#quantizes to 256 values

	#function createData()
	return [X,Y]

data = getData()
print(np.size(data))