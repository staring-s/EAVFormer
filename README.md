# EAVFormer
Our article is under review and we will open source all the code in the future
## Environment configuration
The code is trained andtested with Python=3.10.0, pytorch=1.13.0, torchaudio=0.13.0, opencv-python=4.8.0, scikit-learn=1.3.0, timm=0.9.5, matplotlib=3.7.2.    

## Dataset Download
Download the dataset via the following web link or web disk link. 
- RAVDESS：https://zenodo.org/records/1188976
- CMU-MOSEI：http://immortal.multicomp.cs.cmu.edu/
- CREMA-D: https://github.com/CheyneyComputerScience/CREMA-D

**Note: For the dataset provided in the web links, you need to download the files Video_Speech_Actor_[01-24].zip and Audio_Speech_Actors_01-24.zip. Arrange the data set in the following file structure.**
```
RAVDESS
└───ACTOR01
│   │  01-01-01-01-01-01-01.mp4
│   │  01-01-01-01-01-02-01.mp4
│   │  ...
│   │  03-01-01-01-01-01-01.wav
│   │  03-01-01-01-01-02-01.wav
│   │  ...
└───ACTOR02
└───...
└───ACTOR24
```
