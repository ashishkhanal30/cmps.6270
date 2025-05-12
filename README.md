# CMPS 6270: Deep Learning Project

## Spoken Digit Classification Using  MLP and CNN (Comparision)

This project explores **two different deep learning architectures** for classifying spoken digits (0–7) using the [Free Spoken Digit Dataset (FSDD)](https://github.com/Jakobovski/free-spoken-digit-dataset). It involves:

- **V10 File:** Convolutional Neural Network (CNN) implementation
- **V12 File:** Multilayer Perceptron (MLP) implementation

Each version processes WAV files into spectrograms and evaluates classification accuracy on both dataset samples and real-world custom audio.

---

# V10: Spoken Digit Classification with CNN

## Overview
A Convolutional Neural Network (CNN) was implemented using Keras to classify spoken digits from spectrogram images. The model was evaluated on standard and custom audio recordings.

## Dataset
- Training Data: `recordings/` folder (WAV files at 8kHz)
- Custom Audio: `mySounds/` folder

Each audio file was converted into a 128×128 log-amplitude spectrogram.

## Model Architecture

```
Input: (128, 128, 1) spectrogram
↓ Conv2D (32 filters, 3x3) → ReLU → MaxPool → Dropout
↓ Conv2D (64 filters, 3x3) → ReLU → MaxPool → Dropout
↓ Flatten
↓ Dense (128) → ReLU → Dropout
↓ Dense (8) → Softmax
```

- Optimizer: Adam  
- Loss: Categorical Crossentropy  
- Epochs: 10  
- Train/Validation/Test Split: 72% / 18% / 10%

## Performance Summary

- Test Accuracy: ~96%  
- Accuracy/loss curves and confusion matrices generated

## Spectrogram Visualizations

Visual examples of:
- Training set spectrograms (e.g. `0_george_0`, `1_george_1`, etc.)
- Spectrograms of 8 real-world audio files from `mySounds/`

## Prediction Table (Real-World Samples)

| Audio File        | Actual Class | Predicted Class |
|-------------------|--------------|-----------------|
| 0_george_0.wav    | 0            | 0 (correct)     |
| 1_george_1.wav    | 1            | 2 (incorrect)   |

- Correct: green highlight in table image
- Incorrect: red highlight
- Saved as: `audio_predictions_table_II.png`

## Output Files
- Test Set Confusion Matrix.png
- audio_predictions_table_II.png

---

# V12: Spoken Digit Classification with MLP

## Overview
This version uses a **Multilayer Perceptron (MLP)** model to classify flattened spectrograms instead of treating them as 2D images. The model was evaluated on FSDD dataset.

## Dataset
- Training Data: `recordings/` folder
- Input: Flattened log-amplitude spectrogram of each audio file

## Model Architecture

```
Input: Flattened spectrogram
↓ Dense (512) → ReLU
↓ Dense (512) → Sigmoid
↓ Dense (8) → Softmax
```

- Optimizer: Adam  
- Loss: Categorical Crossentropy  
- Epochs: 10  
- Train/Validation/Test Split: 70% / 15% / 15%

## Performance Summary

- Test Accuracy: ~45%  
- Accuracy/loss curves visualized
- Confusion matrix generated for test set

## Output Files
- Confusion matrix image (MLP)
- Accuracy/loss graphs

---

## Shared Dependencies
```
pip install keras tensorflow librosa matplotlib scikit-learn numpy
```

---

## Author
Project by Aayam Aayam  
Spring 2025  
SELU CMPS 6720
