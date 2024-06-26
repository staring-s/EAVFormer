# CCMA

## Environment configuration
The code is tested with Python=3.9.5, pytorch=1.9.1, numpy=1.22.4, librosa=0.9.1

## Dataset Download
Download the dataset via the following web link or web disk link. 
- RAVDESS：https://zenodo.org/records/1188976
- CMU-MOSEI：http://immortal.multicomp.cs.cmu.edu/
- https://pan.baidu.com/s/1F86Xymx9B13voyquk0LRGA?pwd=grox  

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

## Pre-run procedure
1.The folder 'results' and the file 'EfficientFace_Trained.pth.tar' are found in the following two web disk links. Download them and place them in the same directory as main.py.
- https://drive.google.com/drive/folders/191-YAvH0Q1XdRCHAc3E1_bhfnoBp5BVu?usp=drive_link
- https://pan.baidu.com/s/1F86Xymx9B13voyquk0LRGA?pwd=grox
```
└───models
└───results
└───EfficientFace_Trained.pth.tar
└───...
└───main.py
└───...
```
2.Data preprocessing.Run extract_faces.py, extract_audios.py, and create_annotations.py in ravdess_preprocessing/ folder in sequence to perform data preprocessing and generate annotation files.  
**Note: Before running each .py file, you need to specify the address of the dataset in each .py file.  If the dataset is obtained from a web disk link, run create_annotations.py directly without running extract_faces.py and extract_audios.py.**

## Run
Run the main.py file to execute the project.  
**Note: The relevant parameter Settings are modified in opts.py. The GPU Settings are set at the top of the main.py file.** 
      
## Other
The confusion matrix is generated in Confusion_Matrix.py.   
**Note: The Confusion_Matrix.py file does not run with the project and needs to be run separately after getting the experiment output to generate the confusion matrix. The output for the best experimental effects of this model is given in Confusion_Matrix.py, which can be used to verify the generation of the confusion matrix.** 



