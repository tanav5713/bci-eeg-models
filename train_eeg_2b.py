import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from moabb.datasets import BNCI2014_004
from moabb.paradigms import MotorImagery

from EEGModels import EEGNet
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam

from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from matplotlib import pyplot as plt

K.set_image_data_format('channels_last')

print("Fetching REAL BCI Competition IV 2b Dataset via MOABB...")
print("Note: This will download real EEG data.")
dataset = BNCI2014_004()
dataset.subject_list = [1] # Using Subject 1

# 2b only has 2 classes: left hand, right hand
paradigm = MotorImagery(n_classes=2)

print("Extracting epochs from real data...")
X, labels, metadata = paradigm.get_data(dataset=dataset, subjects=[1])

le = LabelEncoder()
y = le.fit_transform(labels)
names = le.classes_

kernels, chans, samples = 1, X.shape[1], X.shape[2]

split_idx = int(len(X) * 0.8)
X_train      = X[:split_idx]
Y_train      = y[:split_idx]
X_test       = X[split_idx:]
Y_test       = y[split_idx:]

nb_classes = 2
Y_train_cat  = np_utils.to_categorical(Y_train, nb_classes)
Y_test_cat   = np_utils.to_categorical(Y_test, nb_classes)

X_train      = X_train.reshape(X_train.shape[0], chans, samples, kernels)
X_test       = X_test.reshape(X_test.shape[0], chans, samples, kernels)
   
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# Optimized Hyperparameters for BCI IV 2b (3 channels)
# Restored dropout to 0.5 to prevent the network from collapsing into a biased state.
model = EEGNet(nb_classes = nb_classes, Chans = chans, Samples = samples, 
               dropoutRate = 0.5, kernLength = 128, F1 = 8, D = 4, F2 = 32, 
               dropoutType = 'Dropout')

optimizer = Adam(learning_rate=0.001)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, 
              metrics = ['accuracy'])

# Aggressive Manual Class Weights
# The model has a severe bias towards guessing "Left Hand" (Class 0). 
# We will mathematically force it to value "Right Hand" (Class 1) three times more!
class_weights_dict = {0: 1.0, 1: 3.0}

checkpoint_path = 'checkpoint_bci4_2b_real.h5'
checkpointer = ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_best_only=True)
early_stopper = EarlyStopping(monitor='val_accuracy', patience=30, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.0001, verbose=1)

print("---------------------------------------------------------")
print("STARTING TRAINING ON REAL DATA (2b)")
print("---------------------------------------------------------")

fittedModel = model.fit(X_train, Y_train_cat, batch_size = 16, epochs = 150, 
                        verbose = 2, validation_split=0.2,
                        class_weight=class_weights_dict,
                        callbacks=[checkpointer, early_stopper, reduce_lr])

model.load_weights(checkpoint_path)

probs       = model.predict(X_test)
preds       = probs.argmax(axis = -1)  
acc         = np.mean(preds == Y_test)
print("Final EEGNet Classification accuracy on REAL data (2b): %f " % (acc))

print("\n=======================================================")
print("          EEGNET OUTPUT PARAMETERS (METRICS)           ")
print("=======================================================")
print(classification_report(Y_test, preds, target_names=names))
print("=======================================================\n")

print("Training xDAWN + RG model...")
n_components = 2  
clf = make_pipeline(XdawnCovariances(n_components, estimator='oas'),
                    TangentSpace(metric='riemann'),
                    LogisticRegression(C=0.1))

X_train_rg      = X_train.reshape(X_train.shape[0], chans, samples)
X_test_rg       = X_test.reshape(X_test.shape[0], chans, samples)

clf.fit(X_train_rg, Y_train)
preds_rg     = clf.predict(X_test_rg)

acc2         = np.mean(preds_rg == Y_test)
print("Final xDAWN + RG Classification accuracy: %f " % (acc2))

plt.figure(0)
disp = ConfusionMatrixDisplay.from_predictions(Y_test, preds, display_labels=names, cmap=plt.cm.Blues)
disp.ax_.set_title('EEGNet-8,2 (REAL BCI IV 2b)')
plt.savefig('eegnet_real_2b_cm.png')

plt.figure(1)
disp2 = ConfusionMatrixDisplay.from_predictions(Y_test, preds_rg, display_labels=names, cmap=plt.cm.Blues)
disp2.ax_.set_title('xDAWN + RG (REAL BCI IV 2b)')
plt.savefig('xdawn_real_2b_cm.png')

print("Training complete. Models and plots saved.")
