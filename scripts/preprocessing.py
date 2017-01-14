# This script will contain the functions and algorithms for preprocessing the raw audio data into training and testing datasets

# path to folder of sounds
	# get # of sounds
	# iterate over sounds
	# get sample audio data
	# pad sample start with zeros (L = number of samples being fed to base layer)

import os
import wave
import struct
import numpy as np
import network
import random

windowSize = 2048
classes = range(-32768, 32767)
generateSampleRate = 44100
generateChannels = 1
generateLength = 1

trainingBatchSize = 1000
trainingIterationsPerBatch = 10
trainingBatchesBetweenFileGeneration = 10

# mlp classifier is sensitive to feature scaling so we transform inputs to floats from -1 to 1
# 	see: http://scikit-learn.org/stable/modules/neural_networks_supervised.html#tips-on-practical-use
inputScale = 1 / 65536

def getData():
	generatedFileNumber = 1	
	trainingBatchesComplete = 0
	audioFiles = []
	allAudioData = []
	path = '/Users/alex/projects/nn-sampler/scripts/kicks/'
	startFolder = os.listdir(path)

	for file in startFolder:
		#print(file)
		if (file.endswith('.wav')):
			wav = wave.open(path + file)
			waveData = wav.readframes(wav.getnframes())
			audio = struct.unpack("%ih" % (wav.getnframes() * wav.getnchannels()), waveData)
			
			audioFiles.append(list(audio))
			allAudioData += list(audio)

	numFiles = len(audioFiles)
	
	for epoch in range(100):
		random.shuffle(audioFiles)

		X,Y = [],[]

		for file in range(numFiles):
			scaledData = np.asarray(audioFiles[file]) * inputScale
			audioData = (np.zeros(windowSize) - (10000 / inputScale)).tolist() + audioFiles[file]
			totalSamples = len(audioData)
			inputWindow = []

			for t in range(windowSize, totalSamples - 1):
				inputWindow = np.asarray(audioData[t - windowSize:t]) * inputScale
				output = audioData[t] * inputScale

				X.append(inputWindow)
				Y.append(output)

				if (np.shape(Y)[0] >= trainingBatchSize):
					# print('fitting batch, X[0]: ')
					# print(X[0])
					# print('Y[0]: ', Y[0])

					# for n in range(trainingIterationsPerBatch):
					network.mlp.fit(X, Y)
					
					trainingBatchesComplete += 1
					X, Y = [], []

				if(trainingBatchesComplete >= (generatedFileNumber) * trainingBatchesBetweenFileGeneration):
					generatedAudio = (np.zeros(windowSize, dtype=np.int) - (10000 / inputScale)).tolist()
					# generatedAudio = (np.random.rand(windowSize) / inputScale).tolist()

					generatedWavePath = '/Users/alex/projects/nn-sampler/scripts/output/output-' + str(generatedFileNumber) + '.wav'
					generatedWave = wave.open(generatedWavePath, 'w')
					generatedWave.setnchannels(generateChannels)
					generatedWave.setframerate(generateSampleRate)
					generatedWave.setsampwidth(2)

					generatedWaveSampleCount = (generateLength * generateSampleRate)

					for i in range(windowSize, generatedWaveSampleCount + windowSize):
						window = np.asarray(generatedAudio[i - windowSize:i]).reshape(1, -1) * inputScale
						out = network.mlp.predict(window)[0]

						# if(np.mod(i, 64) == 0):
						# 	print('window: ')
						# 	print(window)
						# 	print('output: ', out)
		
						generatedAudio.append(int(out / inputScale))

					try:
						generatedWave.writeframes(struct.pack("%ih" % (generatedWaveSampleCount * generatedWave.getnchannels()), *generatedAudio[windowSize:len(generatedAudio)]))
						print('generated wave file: ', generatedWavePath)
					except struct.error:
						print('struct error while saving file: ', generatedWavePath)

					generatedFileNumber = generatedFileNumber + 1

			print('Finished processing file ', file + 1)


	#print(audioData[0][0])
	#function, grabsound(), that gets a sound from the folder
	#applies 'mu-law companding transformation' 
	#quantizes to 256 values

	#function createData()
	return network.mlp

# data = getData()
# print(np.size(data))
