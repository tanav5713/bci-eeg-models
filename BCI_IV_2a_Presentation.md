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

### Training Terminal Output (Excerpts)
```text
Epoch 108/150
Epoch 108: ReduceLROnPlateau reducing learning rate to 0.0001250000059371814.
12/12 - 3s - 217ms/step - accuracy: 0.8560 - loss: 0.4416 - val_accuracy: 0.8261 - val_loss: 0.5035

Epoch 111/150
12/12 - 2s - 207ms/step - accuracy: 0.8696 - loss: 0.4159 - val_accuracy: 0.8152 - val_loss: 0.5010

Epoch 112/150
12/12 - 3s - 217ms/step - accuracy: 0.8804 - loss: 0.3939 - val_accuracy: 0.8152 - val_loss: 0.5007

Epoch 113/150
Epoch 113: val_loss improved from 0.49885 to 0.49549, saving model to checkpoint_bci4_2a_real.h5
12/12 - 3s - 221ms/step - accuracy: 0.8750 - loss: 0.4187 - val_accuracy: 0.8261 - val_loss: 0.4955

Epoch 114/150
12/12 - 3s - 210ms/step - accuracy: 0.8560 - loss: 0.4455 - val_accuracy: 0.8152 - val_loss: 0.4957

Final EEGNet Classification accuracy on REAL data: 0.862069 
```

## 5. Final Results & Accuracy
The models were evaluated against an unseen 20% validation split. The results indicate exceptional learning capability from the EEGNet architecture.

### EEGNet-8,2 Classification Accuracy
**Accuracy:** `86.20%` 
*(Note: Random chance for 4 classes is 25%, indicating that the model successfully learned complex, generalizable motor imagery patterns).*

### Baseline (xDAWN + RG) Classification Accuracy
**Accuracy:** `75.86%`

### Detailed Class Metrics (EEGNet-8,2)
To verify there are no hidden biases or under-predicted classes, here is the full classification parameter breakdown across all 4 motor imagery targets:

```text
              precision    recall  f1-score   support

        feet       0.81      0.73      0.77        30
   left_hand       0.93      0.93      0.93        30
  right_hand       0.93      0.97      0.95        29
      tongue       0.76      0.81      0.79        27

    accuracy                           0.86       116
   macro avg       0.86      0.86      0.86       116
```

## 6. Confusion Matrices
*The confusion matrices visually demonstrating the class-by-class predictive power can be found below:*

### EEGNet-8,2 Performance
![EEGNet Confusion Matrix](file:///C:/Users/tanav/essex/arl-eegmodels/eegnet_real_cm.png)

### Baseline xDAWN Performance
![xDAWN Confusion Matrix](file:///C:/Users/tanav/essex/arl-eegmodels/xdawn_real_cm.png)
