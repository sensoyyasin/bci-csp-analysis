# bci-csp-lda

This project implements an EEG based Brain-Computer Interface (BCI) pipeline for classifying motor movement and motor imagery tasks using the PhysioNet EEG dataset.

## Overview

The system processes raw EEG signals and classifies epochs into two classes (T1 vs T2) using a machine learning pipeline based on:

- Custom Common Spatial Patterns (CSP)
- Linear Discriminant Analysis (LDA)

## Pipeline

1. Raw EEG loading (EDF format)
2. Channel cleaning and montage assignment
3. ICA-based artifact removal
4. Band-pass filtering (7–30 Hz)
5. Epoch extraction (T1/T2)
6. Motor cortex channel selection
7. CSP feature extraction
8. LDA classification

## Dataset

PhysioNet EEG Motor Movement/Imagery Dataset  
https://physionet.org/content/eegmmidb/1.0.0/

The dataset contains EEG recordings from 109 subjects with 64 channels.

Raw data is not included. Place it in:
physionet.org/

## Usage

Train:

python mybci.py <subject> <experiment> train

Predict:

python mybci.py <subject> <experiment> predict

Summary: 

python mybci.py

Results:

Overall Accuracy        : 0.6841
Overall F1 Score macro  : 0.6840
Overall Precision macro : 0.6842
Overall Recall macro    : 0.6840
Precision-Recall AUC    : 0.7600
ROC AUC                 : 0.7601


The model exceeds the 60% threshold and shows balanced performance across classes.

## Outputs

- Confusion Matrix  
- Precision-Recall Curve  
- ROC Curve  

## Notes

- Offline EEG classification system  
- Performance varies across subjects  
- CSP + LDA provides a simple and efficient baseline  

## References

- PhysioNet EEG Dataset  
- Blankertz et al. (2008) – CSP  
- Gramfort et al. (2013) – MNE  
- Pedregosa et al. (2011) – scikit-learn  
