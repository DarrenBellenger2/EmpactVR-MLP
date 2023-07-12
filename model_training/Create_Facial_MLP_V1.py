import numpy as np
import os, glob, math, random, pickle
import cv2
import dlib
import csv
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
np.random.seed(4)

# ==============================================================================================
# conda activate py37
# cd C:\Users\bellengerd\Documents\PhD\PythonDevelopment\EMPACTVR MLP August2023\model_training
# python Create_Facial_MLP_V1.py
# ==============================================================================================


# Emotions to observe
observed_emotions=['neutral', 'happy', 'sad', 'anger', 'fear', 'disgust', 'surprise']
x_full, y_full = [], []
csv_header = ['Emotion', 'AU1','AU2','AU4','AU6','AU7','AU9','AU15','AU20','AU23','AU25','AU26']
#csv_data_row = [0,0,0,0,0,0,0,0,0,0,0,0]

#Datasets
dataset_name=['CK Train', 'ADFES', 'FEI', 'Jaffe', 'OulaCasia', 'VisGraf']
#dataset_directory=['D:\\Documents\\EmotionDatabases\\Image Libraries\\CK_Train\\%s\\*'] 

#dataset_directory=['c:\\Users\\bellengerd\\Documents\\EmotionDatabases\\Image Libraries\\CK_Labelled\\%s\\*',
#'c:\\Users\\bellengerd\\Documents\\EmotionDatabases\\Image Libraries\\ADFES_Labelled\\%s\\*',
#'c:\\Users\\bellengerd\\Documents\\EmotionDatabases\\Image Libraries\\FEI_Labelled\\%s\\*',
#'c:\\Users\\bellengerd\\Documents\\EmotionDatabases\\Image Libraries\\OulaCasia_Labelled\\%s\\*',
#'c:\\Users\\bellengerd\\Documents\\EmotionDatabases\\Image Libraries\\VisGraf_Labelled\\%s\\*']
#'c:\\Users\\bellengerd\\Documents\\EmotionDatabases\\Image Libraries\\JAFFE_Jpg\\%s\\*',

dataset_directory=[
'D:\\Documents\\EmotionDatabases\\Image Libraries\\CK_Labelled\\%s\\*',
'D:\\Documents\\EmotionDatabases\\Image Libraries\\ADFES_Labelled\\%s\\*',
'D:\\Documents\\EmotionDatabases\\Image Libraries\\FEI_Labelled\\%s\\*',
'D:\\Documents\\EmotionDatabases\\Image Libraries\\JAFFE_Labelled\\%s\\*',
'D:\\Documents\\EmotionDatabases\\Image Libraries\\OulaCasia_Labelled\\%s\\*',
'D:\\Documents\\EmotionDatabases\\Image Libraries\\VisGraf_Labelled\\%s\\*']

rows, cols = (6, 6)
dataset_accuracies = [[0]*cols]*rows

#Set up some required objects
video_capture = cv2.VideoCapture(0) #Webcam object
detector = dlib.get_frontal_face_detector() #Face detector
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #Landmark identifier. Set the filename to whatever 

def distance(From_X, From_Y, To_X, To_Y): 
    dist = math.sqrt( math.pow(To_X - From_X,2) + math.pow(To_Y - From_Y,2) )
    return dist

def lerp(x, a, b): 
    ret = (x - a) / (b - a)
    if ret < 0:
        ret = 0
    return round(ret,2)
    
def extract_feature(image):
    AU1 = 0
    AU2 = 0
    AU4 = 0
    AU6 = 0
    AU7 = 0
    AU9 = 0
    AU15 = 0
    AU20 = 0
    AU23 = 0
    AU25 = 0
    AU26 = 0

    detections = detector(image, 1) 
    for k,d in enumerate(detections): #For all detected face instances individually
        shape = predictor(image, d) #Draw Facial Landmarks with the predictor class
        x = []
        y = []
        for i in range(0,68): #Store X and Y coordinates in two lists
            x.append(float(shape.part(i).x))
            y.append(float(shape.part(i).y))    

    
        # ===============================================================================
        # Evaluate Action Units
		
		
		
        # ===============================================================================
        # AU1: Max = 1.2 & Min = 0.1
        #
        AU1 = (math.atan2(y[17] - y[20], x[20] - x[17]) + math.atan2(y[26] - y[23],x[26] - x[23]))
        AU1 = lerp(AU1,0.1,1.2)
		
		# ===============================================================================
        # AU2: Max = 2.3 & Min = 0.9
        #
        AU2 = (math.atan2(y[17] - y[18], x[18] - x[17]) + math.atan2(y[26] - y[25],x[26] - x[25]))
        AU2 = lerp(AU2,0.9,2.3)
		
		# ===============================================================================
        # AU4: Max = 3 & Min = 1.2
        #
        AU4 = (math.atan2(y[39] - y[21], x[21] - x[39]) + math.atan2(y[42] - y[22],x[42] - x[22]))
        AU4 = lerp(AU4,1.2,3)
        
        # ===============================================================================
        # AU6: Max = 1.16 & Min = 0.7
        #
        AU6 = (distance(x[48],y[48],x[66],y[66]) + distance(x[66],y[66],x[54],y[54])) / (distance(x[48],y[48],x[51],y[51]) + distance(x[51],y[51],x[54],y[54]))
        AU6 = lerp(AU6,0.7,1.16)
        
        # ===============================================================================
        # AU7: Max = 0.5 & Min = 0.05
        #       
        AU7 = (distance(x[37],y[37],x[41],y[41]) + distance(x[62],y[62],x[66],y[66]) + distance(x[63],y[63],x[65],y[65])) / (2 * distance(x[36],y[36],x[45],y[45]))
        AU7 = lerp(AU7,0.05,0.5)

        # ===============================================================================
        # AU9: Max = 2.5 & Min = 1
        #       
        AU9 = (math.atan2(y[50] - y[33], x[33] - x[50]) + math.atan2(y[52] - y[33],x[52] - x[33]))
        AU9 = lerp(AU9,1,2.5)
		
        # ===============================================================================
        # AU15: Max = 1 & Min = -0.5 (-0.8)
        #          
        AU15 = (math.atan2(y[48] - y[60], x[60] - x[48]) + math.atan2(y[54] - y[64],x[54] - x[64]))
        AU15 = lerp(AU15,-0.5,1)

        # ===============================================================================
        # AU20: Max = 3 & Min = 0
        #         
        AU20 = (math.atan2(y[59] - y[65], x[65] - x[59]) + math.atan2(y[55] - y[67],x[55] - x[67]) + math.atan2(y[59] - y[66],x[66] - x[59]) + math.atan2(y[59] - y[67],x[67] - x[59]) + math.atan2(y[55] - y[65],x[55] - x[65]))
        AU20 = lerp(AU20,0,3)

        # ===============================================================================
        # AU23: Max = 9 & Min = 2
        #          
        AU23 = (math.atan2(y[49] - y[50], x[50] - x[49]) + math.atan2(y[53] - y[52],x[53] - x[52]) + math.atan2(y[61] - y[49],x[61] - x[49]) + math.atan2(y[63] - y[53],x[53] - x[63]) + math.atan2(y[58] - y[59],x[58] - x[59]) + math.atan2(y[56] - y[55],x[55] - x[56]) + math.atan2(y[60] - y[51],x[51] - x[60]) + math.atan2(y[64] - y[51],x[64] - x[51]) + math.atan2(y[57] - y[60],x[57] - x[60]) + math.atan2(y[57] - y[64],x[64] - x[57]) + math.atan2(y[62] - y[49],x[62] - x[49]) + math.atan2(y[62] - y[53],x[53] - x[62]) + math.atan2(y[57] - y[60],x[57] - x[60]) + math.atan2(y[57] - y[64],x[64] - x[57]))
        AU23 = lerp(AU23,2,9)
        
        # ===============================================================================
        # AU26: Max = 1 & Min = 0
        #                 
        AU26 = (distance(x[61],y[61],x[67],y[67]) + distance(x[62],y[62],x[66],y[66]) + distance(x[63],y[63],x[65],y[65])) / (distance(x[36],y[36],x[45],y[45]))
        AU26 = lerp(AU26,0,1)		 
        if AU26 > 0.1:
            AU25 = 1
        
        break

    result = np.array([AU1,AU2,AU4,AU6,AU7,AU9,AU15,AU20,AU23,AU25,AU26])   
    return result

def load_data(test_size = 0.2):
    global x_full, y_full
    x, y = [], []
    overall_filecount = 0
    filecount = 0   

    # Changed order from dataset>>emotion TO emotion>>dataset
    with open('output.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        for emotion in observed_emotions:   
            print("Emotion=",emotion)     
            for d_index,d_item in enumerate(dataset_directory):
                print("Dataset=",dataset_name[d_index])     
                filecount = 0       
                for file in glob.glob(d_item %emotion): 
                    print("File=",file)
                    filecount +=1   
                    overall_filecount +=1 
                    
                    image = cv2.imread(file)
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    clahe_image = clahe.apply(gray)

                    detections = detector(clahe_image, 1)   
                    if len(detections) > 0:
                        feature = extract_feature(clahe_image)
                        x.append(feature)
                        y.append(emotion)
                        csv_data_row = feature
                        csv_data_row = np.insert(csv_data_row, 0, observed_emotions.index(emotion), axis=0)
                        writer.writerow(csv_data_row)                       
                print("Filecount=",filecount)               
       
    print("Overall Filecount=",overall_filecount)          
    split = train_test_split(np.array(x), y, test_size = test_size, random_state = 9)
    #split = train_test_split(np.array(x), y, test_size = 0.01, random_state = 9)
    x_full = np.array(x)
    y_full = y
    return split

print("Load data")
x_train, x_test, y_train, y_test = load_data(test_size=0.2)
# Initialize the Multi Layer Perceptron Classifier\n",
#model = MLPClassifier(alpha=0.01,
#                      batch_size=128,
#                      hidden_layer_sizes = (100,200,),
#                      learning_rate='adaptive',
#                      max_iter=500)

# 0.8656903765690377
#model = MLPClassifier(alpha=0.00075,
#                      batch_size=64,
#                      hidden_layer_sizes = (200,100,),
#                      learning_rate='adaptive',
#                      max_iter=500)
   
# 0.997
model = MLPClassifier(alpha=0.00075,
                      batch_size=64,
                      hidden_layer_sizes = (200,100,),
                      learning_rate='adaptive',activation= 'tanh',
                      max_iter=1000)

# 0.947
#model = MLPClassifier(alpha=0.01,
#                      batch_size=64,
#                      hidden_layer_sizes = (200,100,),
#                      learning_rate='adaptive',activation= 'tanh',
#                      max_iter=1000)
 
                      
# Train the model
print("Fit model")
#model.fit(x_train, y_train)
model.fit(x_full, y_full)
                      
print("MLP Classifier")                   

pickle.dump(model, open("..\models\Facial_1.mlp", 'wb'))
              
#y_pred= model.predict(x_test)                     
#accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

y_pred= model.predict(x_full)                     
accuracy = accuracy_score(y_true=y_full, y_pred=y_pred)


#print(\"Accuracy: {:.2f}%\".format(accuracy*100))
print(accuracy)








