# Brain Computer Interface 

This project implements an EEG-based Brain-Computer Interface (BCI) pipeline for classifying motor movement and motor imagery tasks using the PhysioNet EEG dataset.

---

## Electrode Montage

<p align="center">
  <img src="https://github.com/user-attachments/assets/a3a5bf7e-cbf8-49ef-a52a-d9eb66849fb1" height="350"/>
  <img src="https://github.com/user-attachments/assets/c9ea9975-efc4-477a-983e-666ad60c8193" height="350"/>
</p>

<p align="center">
The dataset uses 64 EEG channels based on the international 10–10 system.
</p>

---

## EEG Signals

<p align="center">
  <img src="https://github.com/user-attachments/assets/57cb832d-cce2-4993-96f8-5c1c03acb825" width="75%"/>
</p>

---

## Feature Extraction (CSP)

<p align="center">
  <img src="https://github.com/user-attachments/assets/03391d96-add7-4823-be54-d49aea129dfc" width="45%"/>
  <img src="https://github.com/user-attachments/assets/5d244478-f095-40db-8e10-ca35f376c53a" width="45%"/>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/31282b12-3f55-4325-912f-f31e2419a1c3" width="60%"/>
</p>

---

## Pipeline

<p align="center">
  <img src="https://github.com/user-attachments/assets/05d06b89-0f54-4ec8-b7a8-1ad168d00a20" width="75%"/>
</p>

---

## Overview

The system processes raw EEG signals and classifies epochs into two classes (T1 vs T2) using:

- Custom Common Spatial Patterns (CSP)
- Linear Discriminant Analysis (LDA)

---

## Usage

Train a specific subject and experiment:

    python mybci.py <subject> <experiment> train

Predict a specific subject and experiment:

    python mybci.py <subject> <experiment> predict

Run summary evaluation across all subjects and experiments:

    python mybci.py

Run visualization and testing for a selected subject:

    python visualization.py

The subject can be changed inside `visualization.py` by modifying:

    SUBJECT = "S029"

Run visualization mode to display additional outputs for each experiment:

    python visualization.py --visualize

The `--visualize` option shows raw EEG signals, PSD bands, sensor positions, topomaps, and classification score over time for each experiment.

---

## Evaluation

<p align="center">
  <img src="https://github.com/user-attachments/assets/be647380-8336-47b6-ac9d-ce844a0f3f18" height="350"/>
  <img src="https://github.com/user-attachments/assets/e9c38607-fb24-4642-9b5a-6a1ff88e2dea" height="350"/>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/cb3a6ae0-8a2e-439e-b8d9-6315b1607755" width="60%"/>
</p>
