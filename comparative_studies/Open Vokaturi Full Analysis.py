# OpenVokaWavMean-win64.py
# public-domain sample code by Vokaturi, 2019-05-31
#
# A sample script that uses the VokaturiPlus library to extract the emotions from
# a wav file on disk. The file has to contain a mono recording.
#
# Call syntax:
#   python3 OpenVokaWavMean-win64.py path_to_sound_file.wav
#
# For the sound file hello.wav that comes with OpenVokaturi, the result should be:
#	Neutral: 0.760
#	Happy: 0.000
#	Sad: 0.238
#	Angry: 0.001
#	Fear: 0.000

import glob
import sys
import scipy.io.wavfile

sys.path.append("../api")
import Vokaturi

print ("Loading library...")
Vokaturi.load("../lib/open/win/OpenVokaturi-3-3-win64.dll")
print ("Analyzed by: %s" % Vokaturi.versionAndLicense())

emotions = ["angry", "fearful", "happy", "sad"]
emotion_correct = [0, 0, 0, 0, 0, 0, 0, 0]
emotion_incorrect = [0, 0, 0, 0, 0, 0, 0, 0]
def get_files(emotion):
	files = glob.glob("NEW_RAVDESS_labelled_audio\\%s\\*" %emotion)
	file_list = files[-int(len(files)*1):] #get 100% of file list
	return file_list
	
def predict_emotion(file_name):

	(sample_rate, samples) = scipy.io.wavfile.read(file_name)

	buffer_length = len(samples)
	c_buffer = Vokaturi.SampleArrayC(buffer_length)
	if samples.ndim == 1:  # mono
		c_buffer[:] = samples[:] / 32768.0
	else:  # stereo
		c_buffer[:] = 0.5*(samples[:,0]+0.0+samples[:,1]) / 32768.0

	voice = Vokaturi.Voice (sample_rate, buffer_length)
	voice.fill(buffer_length, c_buffer)

	quality = Vokaturi.Quality()
	emotionProbabilities = Vokaturi.EmotionProbabilities()
	voice.extract(quality, emotionProbabilities)

	emotion = "invalid"
	if quality.valid:
		#print ("Neutral: %.3f" % emotionProbabilities.neutrality)
		#print ("Happy: %.3f" % emotionProbabilities.happiness)
		#print ("Sad: %.3f" % emotionProbabilities.sadness)
		#print ("Angry: %.3f" % emotionProbabilities.anger)
		#print ("Fear: %.3f" % emotionProbabilities.fear)
		
		if emotionProbabilities.neutrality > 0.5:
			emotion = "neutral"
		elif emotionProbabilities.happiness > 0.5:
			emotion = "happy"
		elif emotionProbabilities.sadness > 0.5:
			emotion = "sad"
		elif emotionProbabilities.anger > 0.5:
			emotion = "angry"
		elif emotionProbabilities.fear > 0.5:
			emotion = "fearful"
		
	else:
		#print ("Not enough sonorancy to determine emotions")
		emotion = "none"

	voice.destroy()
	return emotion

for emotion in emotions:
	index = emotions.index(emotion)
	print(" working on %s" %emotion)
	file_list = get_files(emotion)
	for item in file_list:
		predicted_emotion = predict_emotion(item)
		print( "Prediction is", " ", predicted_emotion," -",item)
		if emotion == predicted_emotion:
			emotion_correct[index] += 1
		else:
			emotion_incorrect[index] += 1				
print("== Results ==")
Total_correct = 1
Total_incorrect = 1
for emotion in emotions:
	index = emotions.index(emotion)
	Total_correct += emotion_correct[index]
	Total_incorrect += emotion_incorrect[index]
	emotion_accuracy = emotion_correct[index]
	print("emotion =%s" %emotion)
	print("Correct = %i" %emotion_correct[index])
	print("Incorrect = %i" %emotion_incorrect[index])
Accuracy = Total_correct * (100 / (Total_correct + Total_incorrect))
print("Accuracy = ", Accuracy)	