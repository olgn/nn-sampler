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
generateSampleRate = 44100

trainingBatchSize = 1000
trainingIterationsPerBatch = 10
trainingBatchesBetweenFileGeneration = 10

# mlp classifier is sensitive to feature scaling so we transform inputs to floats from -1 to 1
# 	see: http://scikit-learn.org/stable/modules/neural_networks_supervised.html#tips-on-practical-use
inputScale = 1 / 65534

def getData(path, generateLength = 1):
	generatedFileNumber = 1	
	trainingBatchesComplete = 0
	audioFiles = []
	allAudioData = []
	startFolder = os.listdir(path)
	generateChannels = 0

	for file in startFolder:
		#print(file)
		if (file.endswith('.wav')):
			wav = wave.open(path + file)
			waveData = wav.readframes(wav.getnframes())
			audio = struct.unpack("%ih" % (wav.getnframes() * wav.getnchannels()), waveData)

			if(generateChannels == 0):
				generateChannels = wav.getnchannels()
			
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

					generatedWavePath = path + 'output/output-' + str(generatedFileNumber) + '.wav'
					generatedWave = wave.open(generatedWavePath, 'w')
					generatedWave.setnchannels(generateChannels)
					generatedWave.setframerate(generateSampleRate)
					generatedWave.setsampwidth(2)

					generatedWaveSampleCount = generateLength * generateSampleRate * generateChannels

					for i in range(windowSize, generatedWaveSampleCount + windowSize):
						window = np.asarray(generatedAudio[i - windowSize:i]).reshape(1, -1) * inputScale
						out = network.mlp.predict(window)[0]

						generatedAudio.append(int(out / inputScale))

					try:
						generatedWave.writeframes(struct.pack("%ih" % (generatedWaveSampleCount), *generatedAudio[windowSize:len(generatedAudio)]))
						print('generated wave file: ', generatedWavePath)
					except struct.error as error:
						try:
							generatedAudio = (np.asarray(generatedAudio) * 0.8).astype(int).tolist();
							generatedWave.writeframes(struct.pack("%ih" % (generatedWaveSampleCount), *generatedAudio[windowSize:len(generatedAudio)]))
							print('generated wave file: ', generatedWavePath)
						except struct.error as error:
							print('struct error while saving file ', generatedWavePath, ': ', error)

					generatedFileNumber = generatedFileNumber + 1

			print('Finished processing file ', file + 1)

			# generatedAudio = (np.random.rand(windowSize) / inputScale).tolist()


	#print(audioData[0][0])
	#function, grabsound(), that gets a sound from the folder
	#applies 'mu-law companding transformation' 
	#quantizes to 256 values

	#function createData()
	return network.mlp

# data = getData()
# print(np.size(data))
