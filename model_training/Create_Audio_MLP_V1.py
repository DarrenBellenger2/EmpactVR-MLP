import librosa
import soundfile
from moviepy.editor import *
import numpy as np
import os, glob, pickle
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
np.random.seed(4)

# =====================================================================
# conda activate py37
# cd C:\Users\bellengerd\Documents\PhD\PythonDevelopment\EMPACTVR MLP August2023\model_training
# python Create_Audio_MLP_V1.py
# =====================================================================

# Original accuracy = 0.9518642730321563
# write_audiofile = 0.9096468731505228


# Emotions to observe
observed_emotions=['neutral', 'happy', 'sad', 'anger', 'fear', 'disgust', 'surprise']
x_full, y_full = [], []


# IS TESS getting in the way? Upsampling
#Datasets
#dataset_name=['Shemo', 'SAVEE', 'EmoDB', 'RAVDESS', 'CREMA-D-HI', 'CREMA-D-XX', 'AESDD']
#dataset_name=['Shemo', 'SAVEE', 'EmoDB', 'RAVDESS', 'CREMA-D', 'TESS', 'AESDD', 'ESD', 'URDU', 'Oreau', 'CaFE', 'Emovo', 'JLCorpus', 'SUBESCO', 'MESD']
dataset_name=['Shemo', 'SAVEE', 'EmoDB', 'RAVDESS', 'CREMA-D-HI', 'CREMA-D-XX', 'TESS', 'AESDD', 'Oreau', 'CaFE', 'Emovo', 'JLCorpus', 'SUBESCO', 'MESD', 'ESD-CN', 'URDU'] # ESD,URDU

dataset_directory=['D:\\Documents\\EmotionDatabases\\AudioLibrariesForUse\\Shemo\\combined_labelled\\%s\\*', 
'D:\\Documents\\EmotionDatabases\\AudioLibrariesForUse\\SAVEE\\SAVEE_Audio_Testing\\%s\\*', 
'D:\\Documents\\EmotionDatabases\\AudioLibrariesForUse\\EmoDB\\EMODB_labelled\\%s\\*', 
'D:\\Documents\\EmotionDatabases\\AudioLibrariesForUse\\RAVDESS\\RAVDESS_labelled_audio\\%s\\*', 
'D:\\Documents\\EmotionDatabases\\AudioLibrariesForUse\\CREMA-D\\CREMAD_Labelled_44K\\%s\\*Hi*', 
'D:\\Documents\\EmotionDatabases\\AudioLibrariesForUse\\CREMA-D\\CREMAD_Labelled_44K\\%s\\*Xx*',
'D:\\Documents\\EmotionDatabases\\AudioLibrariesForUse\\Toronto Emotional Speech Set (TESS)\\dataverse_files\\%s\\*',
'D:\Documents\EmotionDatabases\AudioLibrariesForUse\\AESDD\\AESDD_Labelled\\%s\\*',
'D:\Documents\EmotionDatabases\AudioLibrariesForUse\Oreau\Oreau_Labelled\\%s\\*',
'D:\Documents\EmotionDatabases\AudioLibrariesForUse\CaFE\CaFE_Labelled\\%s\\*',
'D:\Documents\EmotionDatabases\AudioLibrariesForUse\Emovo\Emovo_Labelled\\%s\\*',
'D:\Documents\EmotionDatabases\AudioLibrariesForUse\JLCorpus_archive\JLCorpus_Labelled\\%s\\*',
'D:\Documents\EmotionDatabases\AudioLibrariesForUse\SUBESCO\SUBESCO_Labelled\\%s\\*',
'D:\Documents\EmotionDatabases\AudioLibrariesForUse\MESD\MESD_Labelled\\%s\\*',
'D:\Documents\EmotionDatabases\AudioLibrariesForUse\\ESD\\ESD_CN_Labelled\\%s\\*',
'D:\Documents\EmotionDatabases\AudioLibrariesForUse\\URDU\\URDU_Labelled\\%s\\*']





#dataset_directory=['D:\\Documents\\EmotionDatabases\\AudioLibrariesForUse\\Shemo\\combined_labelled\\%s\\*', 
#'D:\\Documents\\EmotionDatabases\\AudioLibrariesForUse\\SAVEE\\SAVEE_Audio_Testing\\%s\\*', 
#'D:\\Documents\\EmotionDatabases\\AudioLibrariesForUse\\EmoDB\\EMODB_labelled\\%s\\*', 
#'D:\\Documents\\EmotionDatabases\\AudioLibrariesForUse\\RAVDESS\\RAVDESS_labelled_audio\\%s\\*', 
#'D:\Documents\EmotionDatabases\AudioLibrariesForUse\\AESDD\\AESDD_Labelled\\%s\\*',
#'D:\\Documents\\EmotionDatabases\\AudioLibrariesForUse\\CREMA-D\\CREMAD_Labelled_44K\\%s\\*Hi*', 
#'D:\\Documents\\EmotionDatabases\\AudioLibrariesForUse\\CREMA-D\\CREMAD_Labelled_44K\\%s\\*Xx*']

#'f:\\Documents\\EmotionDatabases\\AudioLibrariesForUse\\Toronto Emotional Speech Set (TESS)\\dataverse_files\\%s\\*',
#'f:\Documents\EmotionDatabases\AudioLibrariesForUse\\AESDD\\AESDD_Labelled\\%s\\*']


#dataset_directory=['f:\\Documents\\EmotionDatabases\\AudioLibrariesForUse\\Shemo\\combined_labelled\\%s\\*', 
#'f:\\Documents\\EmotionDatabases\\AudioLibrariesForUse\\SAVEE\\SAVEE_Audio_Testing\\%s\\*', 
#'f:\\Documents\\EmotionDatabases\\AudioLibrariesForUse\\EmoDB\\EMODB_labelled_44k\\%s\\*', 
#'f:\\Documents\\EmotionDatabases\\AudioLibrariesForUse\\RAVDESS\\RAVDESS_labelled_audio\\%s\\*', 
#'f:\\Documents\\EmotionDatabases\\AudioLibrariesForUse\\CREMA-D\\CREMAD_Labelled_44k\\%s\\*', 
#'f:\\Documents\\EmotionDatabases\\AudioLibrariesForUse\\Toronto Emotional Speech Set (TESS)\\dataverse_files_44k\\%s\\*',
#'f:\Documents\EmotionDatabases\AudioLibrariesForUse\\AESDD\\AESDD_Labelled\\%s\\*']



#'f:\Documents\EmotionDatabases\AudioLibrariesForUse\\ESD\\ESD_Labelled\\%s\\*',
#'f:\Documents\EmotionDatabases\AudioLibrariesForUse\\URDU\\URDU_Labelled\\%s\\*',
#'f:\Documents\EmotionDatabases\AudioLibrariesForUse\Oreau\Oreau_Labelled\\%s\\*',
#'f:\Documents\EmotionDatabases\AudioLibrariesForUse\CaFE\CaFE_Labelled\\%s\\*',
#'f:\Documents\EmotionDatabases\AudioLibrariesForUse\Emovo\Emovo_Labelled\\%s\\*',
#'f:\Documents\EmotionDatabases\AudioLibrariesForUse\JLCorpus_archive\JLCorpus_Labelled\\%s\\*',
#'f:\Documents\EmotionDatabases\AudioLibrariesForUse\SUBESCO\SUBESCO_Labelled\\%s\\*',
#'f:\Documents\EmotionDatabases\AudioLibrariesForUse\MESD\MESD_Labelled\\%s\\*']
# FULL = 0.9678068410462777 - norm = 0.9295719076740975
# Nomalised MFCC & Chroma = 0.9149899396378269





# +0.25 AESDD
# +15 ESD !!!!!!!!!!
# +2.1 URDU !!!!!!!!!!
# +0.0017  CaFE
# -1.5 Emovo
# +1 JLCorpus
# -2 MESD
# +3.5 SUBESCO !!!!!!!!!!
# -0.5 Oreau

#48K mono 16
# Original 0.7321100917431193
# write_audiofile 0.9518642730321563

#rows, cols = (7, 6)
rows, cols = (15, 6)
dataset_accuracies = [[0]*cols]*rows


#fmin = 125
#fmax = 8000

def extract_feature(file_name, mfcc, chroma, mel):
    #y, sample_rate = librosa.load(librosa.ex('trumpet', hq=True), mono=False)
    #X = librosa.to_mono(y)
    
    
    # ================================================
    # Rubbish
    #clip = AudioFileClip(file_name)
    #clip.write_audiofile("temp.wav", fps=44100, nbytes=4, buffersize=50000, codec='pcm_f32le', ffmpeg_params=["-ac", "1"], verbose=False, logger=None) 
    # ================================================

    y, sample_rate = librosa.load(file_name, sr=44100)    
    X = librosa.to_mono(y)

    if chroma:
        stft = np.abs(librosa.stft(X))
        result = np.array([])
    if mfcc:
        
        # ======================================================================
        # NO normalisation      
        #my_mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        #result = np.hstack((result, my_mfccs))
        
        # ======================================================================
        # Overall normalisation
        full_mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=34)
        subset_mfccs = full_mfccs[1:]
		#[0:13]       
        norm_mfccs = np.subtract(subset_mfccs,np.mean(subset_mfccs))
        my_mfccs = np.mean(norm_mfccs.T, axis=0)            
        result = np.hstack((result, my_mfccs))
 
		# 28 = 0.8946280991735537
		# 36 = 0.8946280991735537
		# 40 = 0.91391

		# 28-1 = 0.9663	
		# 34-1 = 0.9662534435261708
		# 34-1 = 0.9889807162534435 !!!!!!!!!
		# 35-1 = 0.9786501377410468
		# 36-1 = 0.9697	
		# 40-1 = 0.9628
		
		
		# (34-1) on its own = 0.9807162534435262
        
        # ======================================================================
        # Per Bin normalisation
        #full_mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)
        #zero_mfccs = full_mfccs
        #for d_index,d_item in enumerate(full_mfccs):
        #    norm_item = np.subtract(d_item,np.mean(d_item))
        #    zero_mfccs[d_index] = norm_item
        #my_mfccs = np.mean(zero_mfccs.T, axis=0)   
        #result = np.hstack((result, my_mfccs))     
        
    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis = 0)
        #result = np.hstack((result, chroma))
    if mel:
        mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate, n_mels=40).T, axis=0)
        #result = np.hstack((result, mel))

    return result

def load_data(test_size = 0.2):
    global x_full, y_full
    x, y = [], []
    filecount = 0
    
    # Changed order from dataset>>emotion TO emotion>>dataset
    for emotion in observed_emotions:   
        print("==============================")    
        print("Emotion=",emotion)     
        for d_index,d_item in enumerate(dataset_directory):
            print("Dataset=",dataset_name[d_index])     
            filecount = 0       
            for file in glob.glob(d_item %emotion): 
                print("File=",file)
                filecount +=1               
                feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
                x.append(feature)
                y.append(emotion)   
            print("Filecount=",filecount)  
                 
            
    split = train_test_split(np.array(x), y, test_size = test_size, random_state = 9)
    #split = train_test_split(np.array(x), y, test_size = 0.1, random_state = 9)
    x_full = np.array(x)
    y_full = y
    return split

print("Load data")
x_train, x_test, y_train, y_test = load_data(test_size=0.2)
# Initialize the Multi Layer Perceptron Classifier\n",
model = MLPClassifier(alpha=0.01,
                      batch_size=128,
                      hidden_layer_sizes = (100,200,),
                      learning_rate='adaptive',
                      max_iter=500)
                    
# Train the model
print("Fit model")
#model.fit(x_train, y_train)
model.fit(x_full, y_full)
                      
print("MLP Classifier")                   
pickle.dump(model, open("..\models\Audio_1.mlp", 'wb'))
              
y_pred= model.predict(x_test)                     

accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
#print(\"Accuracy: {:.2f}%\".format(accuracy*100))
print(accuracy)







