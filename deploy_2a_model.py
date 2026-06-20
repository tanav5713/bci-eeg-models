"""
=============================================================================
STANDALONE BCI COMPETITION IV 2a INFERENCE SCRIPT
=============================================================================
Instructions for deploying to other platforms:
1. Copy this exact file (`deploy_2a_model.py`) to your new platform.
2. Copy the trained model file (`checkpoint_bci4_2a_real.h5`) to the same directory.
3. Install dependencies: `pip install tensorflow numpy`
4. Run the script! No other files from the repository are required.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

def load_and_test_2a_model(eeg_data_array=None, true_labels=None):
    # 1. Define the parameters
    model_filename = 'checkpoint_bci4_2a_real.h5'
    classes = ['Left Hand', 'Right Hand', 'Both Feet', 'Tongue']
    
    # The exact shape the model requires:
    channels = 22
    samples = 1001
    
    # 2. Check for the model file
    if not os.path.exists(model_filename):
        raise FileNotFoundError(f"Cannot find '{model_filename}'! Please ensure the .h5 model file is in the same directory as this script.")
        
    print(f"Loading 2a Motor Imagery Model from {model_filename}...")
    
    # We suppress compile warnings since we are only doing inference
    model = load_model(model_filename, compile=False)
    print("Model successfully loaded into memory!\n")
    
    # 3. Handle Input Data
    if eeg_data_array is None:
        print("No incoming EEG data detected. Generating a simulated 22-channel trial...")
        # Simulating 1 trial of 22 channels and 1001 time samples
        eeg_data_array = np.random.randn(1, channels, samples, 1)
    
    # 4. Perform Inference
    print("Predicting motor imagery intent...")
    predictions = model.predict(eeg_data_array, verbose=0)
    
    # 5. Output cleanly formatted results
    print("\n" + "="*45)
    print("           REAL-TIME INFERENCE RESULTS")
    print("="*45)
    
    for i, trial_prediction in enumerate(predictions):
        predicted_idx = np.argmax(trial_prediction)
        confidence = np.max(trial_prediction) * 100
        
        print(f"Prediction: {classes[predicted_idx]}")
        print(f"Confidence: {confidence:.2f}%")
        print("\nProbability Breakdown:")
        
        for class_idx, class_name in enumerate(classes):
            prob = trial_prediction[class_idx] * 100
            print(f"  -> {class_name:10s} : {prob:>5.2f}%")
            
    # 6. Calculate Output Parameters (if labels are provided)
    if true_labels is not None:
        try:
            from sklearn.metrics import classification_report
            print("\n" + "="*45)
            print("        MODEL OUTPUT PARAMETERS (METRICS)")
            print("="*45)
            
            # Convert true labels to array if list
            true_labels = np.array(true_labels)
            
            # Ensure predictions match the format
            predicted_classes = np.argmax(predictions, axis=-1)
            
            print(classification_report(true_labels, predicted_classes, target_names=classes))
        except ImportError:
            print("\nWARNING: 'scikit-learn' is required to print the detailed output parameters.")
            print("Install it using: pip install scikit-learn")
            
    print("="*45)

if __name__ == "__main__":
    # To use this in your own project, simply import load_and_test_2a_model 
    # and pass in your real Numpy array of EEG data along with true labels!
    
    # Example Usage:
    # load_and_test_2a_model(eeg_data_array=my_eeg_data, true_labels=[1, 0, 3, 2])
    
    load_and_test_2a_model(eeg_data_array=None, true_labels=None)
