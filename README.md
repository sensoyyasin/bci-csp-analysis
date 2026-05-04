# Brain Computer Interface

This project implements an EEG-based Brain-Computer Interface (BCI) pipeline for classifying motor movement and motor imagery tasks using the PhysioNet EEG dataset.


<img width="451" height="190" alt="image" src="https://github.com/user-attachments/assets/8fcf3cee-ed63-40b8-831d-27fd2a27731a" />


<img width="1776" height="273" alt="image" src="https://github.com/user-attachments/assets/05d06b89-0f54-4ec8-b7a8-1ad168d00a20" />

## Electrodes:

<img width="1316" height="1384" alt="image" src="https://github.com/user-attachments/assets/a3a5bf7e-cbf8-49ef-a52a-d9eb66849fb1" />

<img width="1315" height="983" alt="image" src="https://github.com/user-attachments/assets/c9ea9975-efc4-477a-983e-666ad60c8193" />


## EEG Signals:

<img width="2790" height="1456" alt="image" src="https://github.com/user-attachments/assets/57cb832d-cce2-4993-96f8-5c1c03acb825" />

# Feature Extraction 

<img width="451" height="276" alt="image" src="https://github.com/user-attachments/assets/03391d96-add7-4823-be54-d49aea129dfc" />

<img width="922" height="490" alt="image" src="https://github.com/user-attachments/assets/5d244478-f095-40db-8e10-ca35f376c53a" />

<img width="857" height="735" alt="image" src="https://github.com/user-attachments/assets/31282b12-3f55-4325-912f-f31e2419a1c3" />



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

<img width="1130" height="773" alt="image" src="https://github.com/user-attachments/assets/c745d0f4-6e18-4337-aeaf-48fe96f9f0fb" />


Train:

    python mybci.py <subject> <experiment> train

Predict:

    python mybci.py <subject> <experiment> predict

Summary:

    python mybci.py

<img width="2402" height="804" alt="image" src="https://github.com/user-attachments/assets/0d15d645-6bb8-46e4-94da-22ba5ca184d9" />

<img width="2874" height="1274" alt="image" src="https://github.com/user-attachments/assets/730b5e92-2fd4-49d4-9e31-603eda27624e" />



## Results

    Overall Accuracy        : 0.6841
    Overall F1 Score macro  : 0.6840
    Overall Precision macro : 0.6842
    Overall Recall macro    : 0.6840
    Precision-Recall AUC    : 0.7600
    ROC AUC                 : 0.7601

The model exceeds the 60% threshold and shows balanced performance across classes.

<img width="1425" height="1162" alt="image" src="https://github.com/user-attachments/assets/be647380-8336-47b6-ac9d-ce844a0f3f18" />

<img width="1350" height="970" alt="image" src="https://github.com/user-attachments/assets/e9c38607-fb24-4642-9b5a-6a1ff88e2dea" />

<img width="1288" height="1088" alt="image" src="https://github.com/user-attachments/assets/cb3a6ae0-8a2e-439e-b8d9-6315b1607755" />


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
