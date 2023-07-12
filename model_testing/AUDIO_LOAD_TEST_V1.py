from moviepy.editor import *
from pydub import AudioSegment
from scipy.io import wavfile
import noisereduce as nr
import numpy as np
import os, glob, pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import librosa
import soundfile
np.random.seed(4)
file_count = 0
file_correct = 0
emotion_correct = [0, 0, 0, 0, 0, 0, 0, 0]
emotion_count = [0, 0, 0, 0, 0, 0, 0, 0]

# ==================================================================================
# conda activate py37
# cd C:\Users\bellengerd\Documents\PhD\PythonDevelopment\EMPACTVR MLP August2023\model_testing
# python AUDIO_LOAD_TEST_V1.py
# ==================================================================================

"""
=========================================================
MLP Results - without ESD_EN !
Dataset= RAVDESS MP4   ( 75.48076923076923 %) (files= 1248 )
Dataset= SAVEE AVI     ( 87.5 %) (files= 480 )
Dataset= Shemo        ( 96.14440939362076 %) (files= 2853 )
Dataset= SAVEE        ( 87.70833333333333 %) (files= 480 )
Dataset= EmoDB        ( 90.76923076923077 %) (files= 455 )
Dataset= RAVDESS      ( 91.18589743589743 %) (files= 1248 )
Dataset= CREMA-D-HI   ( 70.64935064935065 %) (files= 385 )
Dataset= CREMA-D-XX   ( 97.51609935602575 %) (files= 1087 )
Dataset= TESS         ( 99.92852037169406 %) (files= 2798 )
Dataset= AESDD        ( 96.84908789386401 %) (files= 603 )
Dataset= Oreau        ( 85.45454545454545 %) (files= 440 )
Dataset= CaFE         ( 90.87301587301587 %) (files= 504 )
Dataset= Emovo        ( 95.578231292517 %) (files= 588 )
Dataset= JLCorpus     ( 97.70833333333333 %) (files= 960 )
Dataset= SUBESCO      ( 95.15714285714286 %) (files= 7000 )

Dataset= ESD-EN   ( 27.113333333333333 %) (files= 15000 )
Dataset= ESD-CN   ( 97.85333333333334 %) (files= 15000 )
Dataset= MESD   ( 97.56380510440835 %) (files= 862 )
Dataset= URDU   ( 99.24812030075188 %) (files= 399 )
=========================================================
"""


# =========================================================
# Emotions to observe
observed_emotions=['neutral', 'happy', 'sad', 'anger', 'fear', 'disgust', 'surprise']
x_check, y_check = [], []

#Datasets

dataset_name=['SAVEE Video']
dataset_directory=['C:\\Users\\bellengerd\\Documents\\EmotionDatabases\\Video Libraries\\SAVEE_Labelled\\%s\\*.avi']

#dataset_name=['Shemo', 'SAVEE', 'EmoDB', 'RAVDESS', 'CREMA-D', 'TESS', 'AESDD', 'ESD', 'URDU', 'Oreau', 'CaFE', 'Emovo', 'JLCorpus', 'SUBESCO', 'MESD']
#dataset_name=['Shemo', 'SAVEE', 'EmoDB', 'RAVDESS', 'CREMA-D', 'TESS', 'AESDD']
#dataset_name=['ESD_EN_Labelled']
#dataset_name=['Shemo', 'SAVEE', 'EmoDB', 'RAVDESS', 'CREMA-D', 'TESS', 'AESDD', 'URDU', 'Oreau', 'CaFE', 'Emovo', 'JLCorpus', 'SUBESCO', 'MESD'] # 'ESD-EN', 'ESD-CN'] 

#dataset_directory=['c:\\users\\bellengerd\\Documents\\EmotionDatabases\\Video Libraries\\RAVDESS_320_Labelled\\%s\\*.mp4']

#dataset_directory=['E:\\Documents\\EmotionDatabases\\Video Libraries\\ADFES_Labelled\\%s\\*.m4v']
#dataset_directory=['E:\\Documents\\EmotionDatabases\\Video Libraries\\Ryerson\\Ryerson_EN_Labelled\\%s\\*.mp4']
#dataset_directory=['D:\\Documents\\EmotionDatabases\\Video Libraries\\eNTERFACE05\\eNTERFACE05_Labelled\\%s\\*.avi']
#dataset_directory=['E:\\Documents\\EmotionDatabases\\Video Libraries\\RAVDESS_Labelled\\%s\\*.mp4']
#dataset_directory=['C:\\Users\\bellengerd\\Documents\\EmotionDatabases\\Video Libraries\\SAVEE_Labelled\\%s\\*.avi']
#dataset_directory=['E:\\Documents\\EmotionDatabases\\Video Libraries\\CREMA-D_Labelled\\%s\\*MD*.mp4']

"""
dataset_name=['RAVDESS MP4', 'SAVEE AVI', 'Shemo', 'SAVEE', 'EmoDB', 'RAVDESS', 'CREMA-D-HI', 'CREMA-D-XX', 'TESS', 'AESDD', 'Oreau', 'CaFE', 'Emovo', 'JLCorpus', 'SUBESCO'] 
dataset_directory=[
'c:\\users\\bellengerd\\Documents\\EmotionDatabases\\Video Libraries\\RAVDESS_320_Labelled\\%s\\*.mp4',
'C:\\Users\\bellengerd\\Documents\\EmotionDatabases\\Video Libraries\\SAVEE_Labelled\\%s\\*.avi',
'D:\\Documents\\EmotionDatabases\\AudioLibrariesForUse\\Shemo\\combined_labelled\\%s\\*', 
'D:\\Documents\\EmotionDatabases\\AudioLibrariesForUse\\SAVEE\\SAVEE_Audio_Testing\\%s\\*', 
'D:\\Documents\\EmotionDatabases\\AudioLibrariesForUse\\EmoDB\\EMODB_labelled\\%s\\*', 
'D:\\Documents\\EmotionDatabases\\AudioLibrariesForUse\\RAVDESS\\RAVDESS_labelled_audio\\%s\\*', 
'D:\\Documents\\EmotionDatabases\\AudioLibrariesForUse\\CREMA-D\\CREMAD_Labelled_44K\\%s\\*Hi*', 
'D:\\Documents\\EmotionDatabases\\AudioLibrariesForUse\\CREMA-D\\CREMAD_Labelled_44K\\%s\\*Xx*',
'D:\\Documents\\EmotionDatabases\\AudioLibrariesForUse\\Toronto Emotional Speech Set (TESS)\\dataverse_files\\%s\\*','D:\Documents\EmotionDatabases\AudioLibrariesForUse\\AESDD\\AESDD_Labelled\\%s\\*',
'D:\Documents\EmotionDatabases\AudioLibrariesForUse\Oreau\Oreau_Labelled\\%s\\*',
'D:\Documents\EmotionDatabases\AudioLibrariesForUse\CaFE\CaFE_Labelled\\%s\\*',
'D:\Documents\EmotionDatabases\AudioLibrariesForUse\Emovo\Emovo_Labelled\\%s\\*',
'D:\Documents\EmotionDatabases\AudioLibrariesForUse\JLCorpus_archive\JLCorpus_Labelled\\%s\\*',
'D:\Documents\EmotionDatabases\AudioLibrariesForUse\SUBESCO\SUBESCO_Labelled\\%s\\*']
"""

"""
dataset_name=['ESD-EN', 'ESD-CN', 'MESD', 'URDU']
dataset_directory=[
'D:\Documents\EmotionDatabases\AudioLibrariesForUse\\ESD\\ESD_EN_Labelled\\%s\\*',
'D:\Documents\EmotionDatabases\AudioLibrariesForUse\\ESD\\ESD_CN_Labelled\\%s\\*',
'D:\Documents\EmotionDatabases\AudioLibrariesForUse\MESD\MESD_Labelled\\%s\\*',
'D:\Documents\EmotionDatabases\AudioLibrariesForUse\\URDU\\URDU_Labelled\\%s\\*']
"""

#dataset_directory=['f:\\Documents\\EmotionDatabases\\AudioLibrariesForUse\\CREMA-D\\CREMAD_Labelled\\%s\\*HI*.wav']


#dataset_directory=['E:\\Documents\\EmotionDatabases\\Video Libraries\\Ryerson\\Ryerson_Converted\\%s\\*.wav']

#dataset_directory=['c:\\Users\\bellengerd\\Documents\\EmotionDatabases\\Video Libraries\\RAVDESS_320_Labelled\\%s\\*.wav']
#dataset_directory=['f:\\Documents\\EmotionDatabases\\AudioLibrariesForUse\\RAVDESS\\RAVDESS_labelled_audio\\%s\\*']

#dataset_directory=['D:\\Documents\\EmotionDatabases\\AudioLibrariesForUse\\Shemo\\combined_labelled\\%s\\*'] 
#dataset_directory=['c:\\Users\\bellengerd\\Documents\\EmotionDatabases\\AudioLibrariesForUse\\SAVEE\\SAVEE_Audio_Testing\\%s\\*'] 
#dataset_directory=['f:\\Documents\\EmotionDatabases\\AudioLibrariesForUse\\EmoDB\\EMODB_labelled\\%s\\*']
#dataset_directory=['f:\\Documents\\EmotionDatabases\\AudioLibrariesForUse\\RAVDESS\\RAVDESS_labelled_audio\\%s\\*']
#dataset_directory=['f:\\Documents\\EmotionDatabases\\AudioLibrariesForUse\\CREMA-D\\CREMAD_Labelled_44k\\%s\\*.wav']
#dataset_directory=['D:\\Documents\\EmotionDatabases\\AudioLibrariesForUse\\Toronto Emotional Speech Set (TESS)\\dataverse_files\\%s\\*']
#dataset_directory=['D:\Documents\EmotionDatabases\AudioLibrariesForUse\\AESDD\\AESDD_Labelled\\%s\\*']
#'f:\Documents\EmotionDatabases\AudioLibrariesForUse\\ESD\\ESD_Labelled\\%s\\*',
#dataset_directory=['D:\Documents\EmotionDatabases\AudioLibrariesForUse\\ESD\\ESD_EN_Labelled\\%s\\*']
#'f:\Documents\EmotionDatabases\AudioLibrariesForUse\\URDU\\URDU_Labelled\\%s\\*',

#dataset_directory=['D:\Documents\EmotionDatabases\AudioLibrariesForUse\Oreau\Oreau_Labelled\\%s\\*']
#'f:\Documents\EmotionDatabases\AudioLibrariesForUse\CaFE\CaFE_Labelled\\%s\\*',
#'f:\Documents\EmotionDatabases\AudioLibrariesForUse\Emovo\Emovo_Labelled\\%s\\*',
#'f:\Documents\EmotionDatabases\AudioLibrariesForUse\JLCorpus_archive\JLCorpus_Labelled\\%s\\*',
#'f:\Documents\EmotionDatabases\AudioLibrariesForUse\SUBESCO\SUBESCO_Labelled\\%s\\*',
#'f:\Documents\EmotionDatabases\AudioLibrariesForUse\MESD\MESD_Labelled\\%s\\*'



#dataset_directory=['f:\\Documents\\EmotionDatabases\\AudioLibrariesForUse\\Shemo\\combined_labelled\\%s\\*', 
#'f:\\Documents\\EmotionDatabases\\AudioLibrariesForUse\\SAVEE\\SAVEE_Audio_Testing\\%s\\*', 
#'f:\\Documents\\EmotionDatabases\\AudioLibrariesForUse\\EmoDB\\EMODB_labelled\\%s\\*', 
#'f:\\Documents\\EmotionDatabases\\AudioLibrariesForUse\\RAVDESS\\RAVDESS_labelled_audio\\%s\\*', 
#'f:\\Documents\\EmotionDatabases\\AudioLibrariesForUse\\CREMA-D\\CREMAD_Labelled_ORIGINAL\\%s\\*', 
#'f:\\Documents\\EmotionDatabases\\AudioLibrariesForUse\\Toronto Emotional Speech Set (TESS)\\dataverse_files\\%s\\*',
#'f:\Documents\EmotionDatabases\AudioLibrariesForUse\\AESDD\\AESDD_Labelled\\%s\\*',
#'f:\Documents\EmotionDatabases\AudioLibrariesForUse\\URDU\\URDU_Labelled\\%s\\*',
#'f:\Documents\EmotionDatabases\AudioLibrariesForUse\Oreau\Oreau_Labelled\\%s\\*',
#dataset_directory=['D:\Documents\EmotionDatabases\AudioLibrariesForUse\CaFE\CaFE_Labelled\\%s\\*']
#'f:\Documents\EmotionDatabases\AudioLibrariesForUse\Emovo\Emovo_Labelled\\%s\\*',
#'f:\Documents\EmotionDatabases\AudioLibrariesForUse\JLCorpus_archive\JLCorpus_Labelled\\%s\\*',
#'f:\Documents\EmotionDatabases\AudioLibrariesForUse\SUBESCO\SUBESCO_Labelled\\%s\\*',
#'f:\Documents\EmotionDatabases\AudioLibrariesForUse\MESD\MESD_Labelled\\%s\\*']

#dataset_directory=['f:\Documents\EmotionDatabases\AudioLibrariesForUse\\ESD\\ESD_EN_Labelled\\%s\\*',
#'f:\Documents\EmotionDatabases\AudioLibrariesForUse\\ESD\\ESD_CN_Labelled\\%s\\*']

#dataset_directory=['f:\Documents\EmotionDatabases\AudioLibrariesForUse\\ESD\\ESD_EN_Labelled\\%s\\*',
#'f:\Documents\EmotionDatabases\AudioLibrariesForUse\\ESD\\ESD_CN_Labelled\\%s\\*']


dataset_accuracies = [[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0]]
dataset_correct = [[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0]]
dataset_count = [[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0]]


print("Load model")
loaded_model = pickle.load(open("..\models\Audio_1.mlp", 'rb'))

def extract_feature(file_name, mfcc, chroma, mel):

    if file_name.endswith('.mp4') or file_name.endswith('.m4v') or file_name.endswith('.avi'):
       
        wav_file = "TEMP.wav"
        clip = AudioFileClip(file_name)
        clip.write_audiofile(wav_file, fps=44100, nbytes=4, buffersize=50000, codec='pcm_f32le', ffmpeg_params=["-ac", "1"], verbose=False, logger=None)

        y, sample_rate = librosa.load(wav_file, sr=44100)
        X = librosa.to_mono(y)
    else:
        y, sample_rate = librosa.load(file_name, sr=44100)
        X = librosa.to_mono(y)
   
        
    if chroma:
        stft = np.abs(librosa.stft(X))
        result = np.array([])
    if mfcc:        
        full_mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=34)
        subset_mfccs = full_mfccs[1:]       
        norm_mfccs = np.subtract(subset_mfccs,np.mean(subset_mfccs))
        my_mfccs = np.mean(norm_mfccs.T, axis=0)            
        result = np.hstack((result, my_mfccs))      
    
    #if chroma:
    #    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis = 0)
    #    #result = np.hstack((result, chroma))
    #if mel:
    #    mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate, n_mels=40).T, axis=0)
    #    #result = np.hstack((result, mel))

    return result

def load_data(test_size = 0.2):
    global x_check, y_check
    global file_count, file_correct
    global emotion_correct, emotion_count   
    x, y = [], []
    sub_count = 0
    sub_correct = 0
    
    for emotion in observed_emotions: 
        index = observed_emotions.index(emotion)    
        print("==============================")    
        print("Emotion=",emotion)   
        for d_index,d_item in enumerate(dataset_directory): 
            sub_count = 0
            sub_correct = 0     
            print("Dataset=",dataset_name[d_index]) 
            for file in glob.glob(d_item %emotion):    
                #print("File=",file)
                audio_data = []
                audio_label = []
                feature = extract_feature(file, mfcc=True, chroma=True, mel=True)           
                x.append(feature)
                y.append(emotion) 
                audio_data.append(feature)  
                audio_label.append(emotion) 
                if len(audio_data) > 0:
                    file_pred = np.array(audio_data)
                    file_pred = file_pred.reshape(1, -1)
                    
                    file_lin = loaded_model.score(file_pred, audio_label)
                    print("Label=%s:", audio_label, " %s:" %file , " %s" %file_lin)
                    
                    index = observed_emotions.index(emotion)                    
                    emotion_correct[index] += file_lin
                    emotion_count[index] += 1                   
                    file_correct += file_lin;
                    file_count += 1
                    sub_count += 1                  
                    sub_correct += file_lin;                    
 

            # Calculate Percentage correct for dataset emotion
            dataset_correct[d_index][index] = 0;   
            dataset_count[d_index][index] = 0;
            dataset_accuracies[d_index][index] = 0;         
            if sub_count > 0:
                dataset_correct[d_index][index] = sub_correct;   
                dataset_count[d_index][index] = sub_count;     
                dataset_accuracies[d_index][index] = ((sub_correct / sub_count) * 100)
            print(d_index,"/",index," Dataset=",dataset_name[d_index], "Correct=",dataset_accuracies[d_index][index],"  (",dataset_count[d_index][index],")") 

 
    x_check = np.array(x)
    y_check = y
    return

print("Load data")
load_data(test_size=0.2)
              
full_pred= loaded_model.predict(x_check)                      
accuracy = accuracy_score(y_true=y_check, y_pred=full_pred)
print("=============================")
print(accuracy)
print("=============================")
print("Single testing")
print(file_count)
print(file_correct)
print((file_correct / file_count) * 100)
print("=============================")

for emotion in observed_emotions:
    index = observed_emotions.index(emotion)
    #Total_correct += emotion_correct[index]
    #Total_incorrect += emotion_count[index] - Total_correct
    emotion_accuracy = 0
    if emotion_correct[index] > 0:
        emotion_accuracy = ((emotion_correct[index] / emotion_count[index]) * 100)
    print("emotion =%s" %emotion)
    print("Correct = %i" %emotion_accuracy)
    #print("Incorrect = %i" %emotion_incorrect[index])


# =========================================================================================================
# Overall - Neutral
cycle_correct=0
cycle_count=0 
print("=============================================")
for d_index,d_item in enumerate(dataset_directory):
    print("=============================================")
    print("Dataset=",dataset_name[d_index])
    cycle_correct=0
    cycle_count=0
    for emotion in observed_emotions:
        e_index = observed_emotions.index(emotion)     
        cycle_correct += dataset_correct[d_index][e_index]
        cycle_count += dataset_count[d_index][e_index]
    cycle_percentage = ((cycle_correct / cycle_count) * 100)
    print("Dataset=",dataset_name[d_index],"  (",cycle_percentage,"%) (files=",cycle_count,")")    
    
    
    
    
    