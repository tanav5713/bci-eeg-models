import os
import sys
import numpy as np
from tensorflow.keras.models import load_model

def run_inference(dataset_type='2a', input_data=None):
    """
    Universal Inference function for the BCI Competition IV EEGNet models.
    
    Args:
        dataset_type (str): Either '2a' or '2b'. 
                            '2a' expects 22 channels (4 classes).
                            '2b' expects 3 channels (2 classes).
        input_data (np.ndarray): Your raw EEG data shaped as (trials, channels, samples, kernels).
                                 If None, a simulated dummy trial will be used to demonstrate.
    """
    if dataset_type not in ['2a', '2b']:
        raise ValueError("Invalid dataset_type. Must be '2a' or '2b'.")

    # 1. Define model paths and configurations
    if dataset_type == '2a':
        model_path = 'checkpoint_bci4_2a_real.h5'
        classes = ['Left Hand', 'Right Hand', 'Both Feet', 'Tongue']
        expected_channels = 22
        expected_samples = 1001 # 4 seconds at 250Hz SMR
    else:
        model_path = 'checkpoint_bci4_2b_real.h5'
        classes = ['Left Hand', 'Right Hand']
        expected_channels = 3
        expected_samples = 1001 # 4 seconds at 250Hz SMR

    # 2. Check if the model file exists locally
    if not os.path.exists(model_path):
        print(f"ERROR: Model file '{model_path}' not found in the current directory.")
        print("Please ensure you have trained the model or placed the .h5 file here.")
        return

    print(f"Loading {dataset_type.upper()} model from {model_path}...")
    model = load_model(model_path)
    print("Model successfully loaded!\n")

    # 3. Handle Input Data
    if input_data is None:
        print(f"No input data provided. Generating a simulated {dataset_type.upper()} EEG trial...")
        # Shape: (trials, channels, time_samples, kernels)
        input_data = np.random.randn(1, expected_channels, expected_samples, 1)
    
    # Validation
    if len(input_data.shape) != 4 or input_data.shape[1] != expected_channels:
        print(f"WARNING: Your input data shape {input_data.shape} does not match the expected format.")
        print(f"Expected format for {dataset_type}: (number_of_trials, {expected_channels}, samples, 1)")
        print("Attempting to run prediction anyway...\n")

    # 4. Run Prediction
    print("Running inference...")
    predictions = model.predict(input_data)

    # 5. Output Results
    print("\n" + "="*40)
    print(f"      INFERENCE RESULTS ({dataset_type.upper()} MODEL)")
    print("="*40)
    
    for i in range(len(input_data)):
        pred_idx = np.argmax(predictions[i], axis=-1)
        confidence = np.max(predictions[i]) * 100
        print(f"Trial {i+1}:")
        print(f"  Predicted Intent: {classes[pred_idx]}")
        print(f"  Confidence:       {confidence:.2f}%")
        
        # Show breakdown across all classes
        print("  Class Breakdown:")
        for j, cls in enumerate(classes):
            print(f"    - {cls}: {predictions[i][j]*100:.1f}%")
        print("-" * 40)

if __name__ == '__main__':
    # You can easily change this to '2b' to test the other model!
    print("Testing BCI Competition IV 2a Model:")
    run_inference(dataset_type='2a', input_data=None)

    print("\n\nTesting BCI Competition IV 2b Model:")
    run_inference(dataset_type='2b', input_data=None)
