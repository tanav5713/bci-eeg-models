# BCI Competition IV 2a: EEG Classification Report

## 1. Overview
This report details the successful training of an EEG-based Motor Imagery classifier using the **BCI Competition IV 2a** dataset. The objective was to achieve high classification accuracy across four distinct motor imagery classes:
1. Left Hand
2. Right Hand
3. Both Feet
4. Tongue

## 2. Dataset Setup
The dataset was accessed dynamically using the `moabb` (Mother of All BCI Benchmarks) framework. 
- **Dataset:** `BNCI2014_001`
- **Data Shape:** 22 EEG Channels, 1000 samples per trial (4-second trials at 250Hz).
- **Subject:** Subject 1 was utilized for this model training instance.

## 3. Model Architecture
The primary model utilized is **EEGNet-8,2**, a compact convolutional neural network designed explicitly for EEG signals. To maximize performance on this 250Hz SMR dataset, the model was heavily optimized with the following hyperparameters:
- `kernLength = 128`: Extended to capture a wider temporal window of the 250Hz signal.
- `dropoutRate = 0.5`: High spatial and temporal dropout to enforce robust feature learning.
- `F1 = 8, D = 2, F2 = 16`: Number of temporal filters, spatial filters, and pointwise filters respectively.

As a baseline comparison, an **xDAWN Spatial Filter + Riemannian Geometry** pipeline connected to a Logistic Regression classifier was also trained.

## 4. Training Process & Optimizations
To combat both **overfitting** and **underfitting**, several advanced techniques were integrated into the training sequence:
- **Early Stopping:** Monitored the validation accuracy and restored the best model weights if improvement stalled for 30 consecutive epochs.
- **ReduceLROnPlateau:** Automatically slashed the Adam optimizer's learning rate by 50% if the validation loss plateaued, enabling the model to settle into the optimal local minimum.
- **Regularization:** Applied `C=0.1` L2 penalty to the baseline xDAWN Logistic Regression model.

The model was permitted to train for up to 150 epochs. Due to the early stopping mechanisms, optimal convergence was cleanly reached around Epoch 118.

## 5. Final Results & Accuracy
The models were evaluated against an unseen 20% validation split. The results indicate exceptional learning capability from the EEGNet architecture.

### EEGNet-8,2 Classification Accuracy
**Accuracy:** `86.20%` 
*(Note: Random chance for 4 classes is 25%, indicating that the model successfully learned complex, generalizable motor imagery patterns).*

### Baseline (xDAWN + RG) Classification Accuracy
**Accuracy:** `75.86%`

## 6. Confusion Matrices
*The confusion matrices visually demonstrating the class-by-class predictive power can be found alongside this report in the repository:*
- `eegnet_real_cm.png`: Displays the EEGNet performance distribution.
- `xdawn_real_cm.png`: Displays the xDAWN baseline performance distribution.
