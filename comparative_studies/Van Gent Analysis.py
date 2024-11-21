import glob
import random
import math
import numpy as np
import cv2
import dlib
import itertools
import pickle
from sklearn.svm import SVC
emotions = ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"] #Emotion list
emotion_correct = [0, 0, 0, 0, 0, 0, 0, 0]
emotion_incorrect = [0, 0, 0, 0, 0, 0, 0, 0]
data = {} #Make dictionary for all values
clf = pickle.load(open('trained_model.sav', 'rb'))	
#Set up some required objects
video_capture = cv2.VideoCapture(0) #Webcam object
detector = dlib.get_frontal_face_detector() #Face detector
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #Landmark identifier. Set the filename to whatever you named the downloaded file
def get_files(emotion): #Define function to get file list, randomly shuffle it and split 80/20
	files = glob.glob("CKdataset\\%s\\*" %emotion)
	random.shuffle(files)
	prediction = files[-int(len(files)*1):] #get 100% of file list
	return prediction
def get_landmarks(image):
	detections = detector(image, 1)
	for k,d in enumerate(detections): #For all detected face instances individually
		shape = predictor(image, d) #Draw Facial Landmarks with the predictor class
		xlist = []
		ylist = []
		for i in range(1,68): #Store X and Y coordinates in two lists
			xlist.append(float(shape.part(i).x))
			ylist.append(float(shape.part(i).y))
		xmean = np.mean(xlist)
		ymean = np.mean(ylist)
		xcentral = [(x-xmean) for x in xlist]
		ycentral = [(y-ymean) for y in ylist]
		landmarks_vectorised = []
		for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
			landmarks_vectorised.append(w)
			landmarks_vectorised.append(z)
			meannp = np.asarray((ymean,xmean))
			coornp = np.asarray((z,w))
			dist = np.linalg.norm(coornp-meannp)
			landmarks_vectorised.append(dist)
			landmarks_vectorised.append((math.atan2(y, x)*360)/(2*math.pi))
		data['landmarks_vectorised'] = landmarks_vectorised
	if len(detections) < 1:
		data['landmarks_vestorised'] = "error"
for emotion in emotions:
	index = emotions.index(emotion)
	print(" working on %s" %emotion)
	prediction = get_files(emotion)
	for item in prediction:
		image = cv2.imread(item)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
		clahe_image = clahe.apply(gray)
		get_landmarks(clahe_image)
		if data['landmarks_vectorised'] == "error":
			print("no face detected on this one")
		else:
			np_array = np.array(data['landmarks_vectorised']) #Turn the training set into a numpy array for the classifier		
			emo_val = clf.predict(np_array.reshape(1,-1))
			emo_string = emotions[int(emo_val)]
			if emotion == emo_string:
				emotion_correct[index] += 1
			else:
				emotion_incorrect[index] += 1
		if cv2.waitKey(1) & 0xFF == ord('q'): #Exit program when the user presses 'q'
			break
print("== Results ==")
Total_correct = 0
Total_incorrect = 0
for emotion in emotions:
	index = emotions.index(emotion)
	Total_correct += emotion_correct[index]
	Total_incorrect += emotion_incorrect[index]
Accuracy = Total_correct * (100 / (Total_correct + Total_incorrect))
print("Accuracy = %i percent" %Accuracy)	