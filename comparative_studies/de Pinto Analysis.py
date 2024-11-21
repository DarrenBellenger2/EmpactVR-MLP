import glob
import keras
import numpy as np
import librosa
#emotions = ["angry", "disgust", "fearful", "happy", "sad", "surprised"]
emotions = ["surprised"]
emotion_correct = [0, 0, 0, 0, 0, 0, 0, 0]
emotion_incorrect = [0, 0, 0, 0, 0, 0, 0, 0]

def get_files(emotion):
	files = glob.glob("RAVDESS_labelled_audio\\%s\\*" %emotion)
	file_list = files[-int(len(files)*1):] #get 100% of file list
	return file_list
class livePredictions:

    def __init__(self, path, file):

        self.path = path
        self.file = file

    def load_model(self):
        '''
        I am here to load you model.

        :param path: path to your h5 model.
        :return: summary of the model with the .summary() function.

        '''
        self.loaded_model = keras.models.load_model(self.path)
        #return self.loaded_model.summary()
        return

    def makepredictions(self):
        data, sampling_rate = librosa.load(self.file)
        mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
        x = np.expand_dims(mfccs, axis=2)
        x = np.expand_dims(x, axis=0)
        predictions = self.loaded_model.predict_classes(x)	
        print( "Prediction is", " ", self.convertclasstoemotion(predictions)," -",self.file, "emotion=",emotion)
        if emotion == predictions:
            emotion_correct[index] += 1
        if emotion != predictions:
            emotion_incorrect[index] += 1

    def convertclasstoemotion(self, pred):
        '''
        I am here to convert the predictions (int) into human readable strings.
        '''
        self.pred  = pred

        if pred == 0:
            pred = "neutral"
            return pred
        elif pred == 1:
            pred = "calm"
            return pred
        elif pred == 2:
            pred = "happy"
            return pred
        elif pred == 3:
            pred = "sad"
            return pred
        elif pred == 4:
            pred = "angry"
            return pred
        elif pred == 5:
            pred = "fearful"
            return pred
        elif pred == 6:
            pred = "disgust"
            return pred
        elif pred == 7:
            pred = "surprised"
            return pred

# Here you can replace path and file with the path of your model and of the file from the RAVDESS dataset you want to use for the prediction,
# Below, I have used a neutral file: the prediction made is neutral.

for emotion in emotions:
	index = emotions.index(emotion)
	print(" working on %s" %emotion)
	file_list = get_files(emotion)
	for item in file_list:
		#print(" working on file %s" %item)
		pred = livePredictions(path='d:/Documents/Audio Emotion Research/Emotion-Classification-Ravdess-master/Emotion_Voice_Detection_Model.h5',file=item)
		pred.load_model()
		pred.makepredictions()
		#if emotion == predicted_emotion:
		#	emotion_correct[index] += 1
		#else:
		#	emotion_incorrect[index] += 1		
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
print("Accuracy = %i percent" %Accuracy)