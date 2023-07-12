# EmpactVR-MLP

EMPACTVR is a research project to integrate bi-modal (facial and audio) emotion recognition, into a metaverse or game platform. This is the first version of EMPACTVR, a baseline of 2 MLP machine learning modules utlised within a Python webcam demo to provide an early look at the development. Later a Unity-specific version will be created, that can be integrated into a metaverse or game. 

![alt text](https://github.com/DarrenBellenger2/EmpactVR-MLP/blob/main/paper/PrototypeOne.jpg)

# Facial Emotion Recognition

It is theorised that facial values present in some types of avatars (such as the Unity UMA avatar) resemble Ekman action units. The early results presented show a facial emotion recognition accuracy of up to 92% on one benchmark dataset, with an overall accuracy of 77.2% across a wide range of datasets, demonstrating the early promise of the research.

This [paper](https://github.com/DarrenBellenger2/EmpactVR-MLP/blob/main/paper/Darren_Bellenger_Wiley_Positional_Paper___Facial_Emotion_Recognition_6.pdf) outlines the creation of a feature extractor that produces facial settings that could drive an avatar (UMA) along with provide values for an ML model.

The datasets utlised are: [CK+](https://paperswithcode.com/dataset/ck), [Helen](http://www.ifp.illinois.edu/~vuongle2/helen/), [Yale](https://www.kaggle.com/datasets/olgabelitskaya/yale-face-database), [Jaffe](https://paperswithcode.com/dataset/jaffe), [VisGraf](https://www.visgraf.impa.br/t-faces/index.html), [ADFES](https://aice.uva.nl/research-tools/adfes-stimulus-set/adfes-stimulus-set.html?cb), [Oulu-Casia](https://paperswithcode.com/dataset/oulu-casia).

# Audio Emotion Recognition

The audio emotion recognition phase is performed by analysing the mel-frequency cepstrum (MFC), using this as the sole feature input into an ML model. The model is able to recognise emotion in full sentences and smaller parts of a sentence. The diverse set of multi-langaue datasets utlised are: [Shemo](https://paperswithcode.com/dataset/shemo), [SAVEE](http://kahlan.eps.surrey.ac.uk/savee/), [EmoDB](https://www.kaggle.com/datasets/piyushagni5/berlin-database-of-emotional-speech-emodb), [RAVDESS](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio), [CREMA-D](https://www.kaggle.com/datasets/ejlok1/cremad), [TESS](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess), [AESDD](http://m3c.web.auth.gr/research/aesdd-speech-emotion-recognition/), [Oreau](https://zenodo.org/record/4405783), [CaFE](https://zenodo.org/record/1478765), [Emovo](https://paperswithcode.com/dataset/emovo), [JLCorpus](https://www.kaggle.com/datasets/tli725/jl-corpus), [SUBESCO](https://zenodo.org/record/4526477), [MESD](https://www.kaggle.com/datasets/saurabhshahane/mexican-emotional-speech-database-mesd), [ESD](https://paperswithcode.com/dataset/esd), [URDU](https://www.kaggle.com/datasets/kingabzpro/urdu-emotion-dataset).


