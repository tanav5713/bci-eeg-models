import numpy as np
import warnings
warnings.filterwarnings('ignore')
from moabb.datasets import BNCI2014_004
from moabb.paradigms import MotorImagery
from tensorflow.keras.models import load_model
from tensorflow.keras import utils as np_utils
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import os

print("Evaluating 2a Model...")
dataset = BNCI2014_004()
dataset.subject_list = [1]
paradigm = MotorImagery(n_classes=4)
X, labels, metadata = paradigm.get_data(dataset=dataset, subjects=[1])
le = LabelEncoder()
y = le.fit_transform(labels)
names = le.classes_
kernels, chans, samples = 1, X.shape[1], X.shape[2]
split_idx = int(len(X) * 0.8)
X_test = X[split_idx:]
Y_test = y[split_idx:]
X_test = X_test.reshape(X_test.shape[0], chans, samples, kernels)
model = load_model('checkpoint_bci4_2a_real.h5', compile=False)
probs = model.predict(X_test, verbose=0)
preds = probs.argmax(axis=-1)
print(classification_report(Y_test, preds, target_names=names))
