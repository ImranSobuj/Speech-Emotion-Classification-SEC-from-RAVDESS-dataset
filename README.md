#Speech-Analysis
##Requirements
- **Python 3.6+**
### Python Packages
librosa==0.8.0
optuna==2.7.0
numpy==1.19.5
pandas
soundfile==0.9.0
wave
sklearn
tqdm==4.28.1
matplotlib==2.2.3
pyaudio==0.2.11

Install these libraries by the following command:
```
pip3 install -r requirements.txt

```
Run all the codes sequentially cell by cell in colab

#Abstruct


This notebook includes code for reading in audio data, feature extraction, hyperparameter tuning with Optuna, and models including KNN, Logistic Regression, Bagging, Multilayer Perceptron. The Python library libROSA provided the main tools for processing and extracting features from the audio files utilized in this project.  

Beginning with extracting MFCCs, Chroma, and Mel spectrograms from the audio files modeling were done with readily available models from Sci-kit Learn and other Python packages. Hyperparameter tuning for these models was accomplished using the Optuna framework.

## Introduction 

Classifying audio to emotion is challenging because of its subjective nature. This task can be challenging for humans, let alone machines. Potential applications for classifying audio to emotion are numerous, including call centers, AI assistants, counseling, and veracity tests.  

There are numerous projects and articles available on this subject. Please see the references section at the bottom of this readme for useful and interesting articles and Jupyter notebooks on this or related topics.

Overview of this notebook's approach for classifying audio to emotion:
- Read WAV files by using the libROSA package in Python.
- Extract features from the audio time series using functions from the libROSA package (MFCCs, Chroma, and Mel spectrograms).
- Construct a series of models from various readily available Python packages.
- Tune hyperparameters for the models using the Optuna framework.
- Ensemble models using a soft voting classifier to improve performance.

Audio is represented as waves where the x-axis is time and the y-axis is amplitude.  These waves are stored as a sum of sine waves using three values as in *A* sin(*B*t +*C*), where *A* controls the amplitude of the curve, *B* controls the period of the curve, and *C* controls the horizontal shift of the curve.  Samples are recorded at every timestep, and the number of samples per second is called the sampling rate, typically measured in hertz (Hz), which are defined as cycles per one second.  The standard sampling rate in libROSA is 22,050 Hz because that is the upper bound of human hearing.

### Data Description
Each of the 7356 RAVDESS files has a unique filename. The filename consists of a 7-part numerical identifier (e.g., 02-01-06-01-02-01-12.mp4). These identifiers define the stimulus characteristics:

Filename identifiers
Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
Vocal channel (01 = speech, 02 = song).
Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the ‘neutral’ emotion.
Statement (01 = “Kids are talking by the door”, 02 = “Dogs are sitting by the door”).
Repetition (01 = 1st repetition, 02 = 2nd repetition).
Actor (01 to 24. Odd-numbered actors are male, even-numbered actors are female).

Filename example: 02-01-06-01-02-01-12.mp4

Video-only (02)
Speech (01)
Fearful (06)
Normal intensity (01)
Statement “dogs” (02)
1st Repetition (01)
12th Actor (12)
Female, as the actor ID number is even.

The dataset is composed of 24 professional actors, 12 male and 12 female making it gender-balanced. 

The audio files were created in a controlled environment and using identical statements spoken in an American accent. 
- Speech file (Audio_Speech_Actors_01-24.zip, 215 MB) contains 1440 files: 60 trials per actor x 24 actors = 1440. 
-
The files are in the WAV raw audio file format and all have a 16-bit Bitrate and a 48 kHz sample rate. The files are all uncompressed, lossless audio, and have not lost any information/data or been modified from the original recording. 

## Feature Extraction

As mentioned before, the audio files were processed using the libROSA python package. This package was created for music and audio analysis, making it a good selection. After importing libROSA, the WAV files are read one at a time. An audio time series in the form of a 1-dimensional array for mono or 2-dimensional array for stereo, along with time sampling rate (also defines the length of the array), where the elements within each of the arrays represent the amplitude of the sound waves is returned by libROSA’s “load” function.

Some helpful definitions for understanding the features used:

- **Mel scale** — deals with human perception of frequency, it is a scale of pitches judged by listeners to be equal distance from each other
- **Pitch** — how high or low a sound is. It depends on frequency, higher pitch is high frequency
- **Frequency** — speed of vibration of sound, measures wave cycles per second
- **Chroma** — Representation for audio where spectrum is projected onto 12 bins representing the 12 distinct semitones (or chroma). Computed by summing the log frequency magnitude spectrum across octaves.
- **Fourier Transforms** — used to convert from the time domain to frequency domain
  - *time-domain*: shows how signal changes over time
  - *frequency domain*: shows how much of the signal lies within each given frequency band over a range of frequencies

Using the signal extracted from the raw audio file and several of libROSA’s audio processing functions, MFCCs, Chroma, and Mel spectrograms were extracted using a function that receives a file name (path), loads the audio file, then utilizes several libROSA functions to extract features that are then aggregated and returned in the form of a numpy array.

### Summary of Features

- **MFCC** - Mel Frequency Cepstral Coefficients: 
Voice is dependent on the shape of the vocal tract including the tongue, teeth, etc.
Representation of short-time power spectrum of sound, essentially a representation of the vocal tract

- **STFT** - returns complex-valued matrix D of short-time Fourier Transform Coefficients:
Using abs(D[f,t]) returns the magnitude of frequency bin f at frame t (Used as an input for Chroma_STFT)

- **Chroma_STFT** - (12 pitch classes) using an energy (magnitude) spectrum (obtained by taking the absolute value of the matrix returned by libROSA’s STFT function) instead of power spectrum returns normalized energy for each chroma bin at each frame

- **Mel Spectrogram** - magnitude spectrogram computed then mapped onto mel scale—the x-axis is time, the y-axis is the frequency

## Modeling

After all of the files were individually processed through feature extraction, the dataset was split into an 80% train set and 20% test set. This split size can be adjusted in the data loading function.

A Breakdown of the Models:
- Simple models: K-Nearest Neighbors, Logistic Regression
- Ensemble models: Bagging (Random Forest)
- Multilayer Perceptron Classifier


The hyperparameters for each of the models above were tuned with the Optuna framework using the mean accuracy of 5-fold cross-validation on the train set as the metric to optimize. This particular framework was chosen due to its flexibility, as it allows for distributions of numerical values or lists of categories to be suggested for each of the hyperparameters. It's the pruning of unpromising trials that makes it faster than a traditional grid search.

## Results
The results and parameters of the top-performing models are provided below, as well as a summary of metrics obtained by other models. Note that results will vary slightly with each run of the associated Jupyter notebooks unless seeds are set.  

### Multilayer Perceptron
MLP is the best ML algorithm for classifying neutral, positive, and negative emotions without cross-validation. 

Test Stats
               precision    recall  f1-score   support

    negative       0.86      0.83      0.84       182
     neutral       0.84      0.86      0.85        65
    positive       0.51      0.56      0.53        41

    accuracy                           0.80       288
   macro avg       0.73      0.75      0.74       288
weighted avg       0.80      0.80      0.80       288

Random Forest Classifier

Test Stats
               precision    recall  f1-score   support

    negative       0.77      0.90      0.83       182
     neutral       0.75      0.74      0.74        65
    positive       0.69      0.22      0.33        41

    accuracy                           0.76       288
   macro avg       0.74      0.62      0.64       288
weighted avg       0.76      0.76      0.74       288

K-Nearest Neighbors

Test Stats
               precision    recall  f1-score   support

    negative       0.73      0.92      0.81       182
     neutral       0.85      0.62      0.71        65
    positive       0.38      0.12      0.19        41

    accuracy                           0.74       288
   macro avg       0.66      0.55      0.57       288
weighted avg       0.71      0.74      0.70       288

Logistic Regression

Test Stats
               precision    recall  f1-score   support

    negative       0.75      0.75      0.75       182
     neutral       0.65      0.69      0.67        65
    positive       0.38      0.34      0.36        41

    accuracy                           0.68       288
   macro avg       0.59      0.59      0.59       288
weighted avg       0.67      0.68      0.67       288
  

MLP is the best ML algorithm for classifying neutral, positive, and negative emotions with cross-validation. 


MLP Classifier

Test Stats
               precision    recall  f1-score   support

    negative       0.84      0.82      0.83       182
     neutral       0.81      0.85      0.83        65
    positive       0.51      0.54      0.52        41

    accuracy                           0.78       288
   macro avg       0.72      0.73      0.73       288
weighted avg       0.79      0.78      0.79       288





## Conclusion

The use of three features (MFCC’s, Mel Spectrograms, and chroma STFT) gave impressive accuracy in most of the models, reiterating the importance of feature selection.  As with many data science projects, different features could be used and/or engineering.  Tonnetz was originally used in modeling, however, it led to decreased performance and was removed. Some other possible features to explore concerning audio would be MFCC Filterbanks or features extracted using the perceptual linear predictive (PLP) technique.  These features could affect the performance of models in the emotion classification task.  

## Future Work
We aim to use Wav2vec, a State-of-the-art speech recognition through self-supervision in the future. It would be interesting to see how a human classifying the audio would measure up to these models, however, finding someone willing to listen to more than 2,400 audio clips may be a challenge in itself because a person can only listen to “the children are talking by the door” or “the dogs are sitting by the door” so many times.

## References
https://zenodo.org/record/1188976#.XeqDKej0mMo  
http://conference.scipy.org/proceedings/scipy2015/pdfs/brian_mcfee.pdf  
https://towardsdatascience.com/ok-google-how-to-do-speech-recognition-f77b5d7cbe0b  
http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/  
https://en.wikipedia.org/wiki/Frequency_domain  
http://www.nyu.edu/classes/bello/MIR_files/tonality.pdf  
https://github.com/marcogdepinto/Emotion-Classification-Ravdess/blob/master/EmotionsRecognition.ipynb  
https://towardsdatascience.com/speech-emotion-recognition-with-convolution-neural-network-1e6bb7130ce3  
https://data-flair.training/blogs/python-mini-project-speech-emotion-recognition/  
https://labrosa.ee.columbia.edu/matlab/chroma-ansyn/  
https://librosa.github.io/librosa/index.html  

