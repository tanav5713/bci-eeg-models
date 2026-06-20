import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from moabb.datasets import BNCI2014_004
from moabb.paradigms import MotorImagery

from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

print("Loading cached BCI Competition IV 2b dataset...")
dataset = BNCI2014_004()
dataset.subject_list = [1] 

paradigm = MotorImagery(n_classes=2)

# This will be instant because the dataset was cached on the disk during the previous run!
X, labels, metadata = paradigm.get_data(dataset=dataset, subjects=[1])

le = LabelEncoder()
y = le.fit_transform(labels)
names = le.classes_

kernels, chans, samples = 1, X.shape[1], X.shape[2]

split_idx = int(len(X) * 0.8)
X_test = X[split_idx:]
Y_test = y[split_idx:]
X_test = X_test.reshape(X_test.shape[0], chans, samples, kernels)

print("\nLoading trained EEGNet model...")
model_path = 'checkpoint_bci4_2b_real.h5'
model = load_model(model_path)

print("Running predictions on the unseen test split...")
probs = model.predict(X_test)
preds = probs.argmax(axis = -1)  

print("\n-----------------------------------------------------------")
print("DETAILED PERFORMANCE METRICS (Precision, Recall, F1-Score)")
print("-----------------------------------------------------------")
report = classification_report(Y_test, preds, target_names=names)
print(report)
