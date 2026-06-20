import numpy as np
from tensorflow.keras.models import load_model

# ---------------------------------------------------------
# BCI Competition IV 2a Inference Example
# ---------------------------------------------------------
# This script demonstrates how to load your trained model
# into a new project and test it on incoming EEG data.

# 1. Path to your saved model file
model_path = 'checkpoint_bci4_2a_real.h5'

print(f"Loading trained model from {model_path}...")
model = load_model(model_path)
print("Model loaded successfully!")

# 2. Define the class labels used during training
classes = ['Left Hand', 'Right Hand', 'Both Feet', 'Tongue']

# 3. Simulate incoming EEG data
# For the 2a model, the input shape must be:
# (number_of_trials, channels, samples, kernels)
# i.e., (1, 22, 1001, 1) for a single 4-second trial (from t=0s to t=4.0s inclusive)
print("\nSimulating a single EEG trial (22 channels, 1001 samples)...")
dummy_trial = np.random.randn(1, 22, 1001, 1)

# 4. Run Inference (Testing the model)
print("Running prediction...")
predictions = model.predict(dummy_trial)

# 5. Extract the results
predicted_class_idx = np.argmax(predictions, axis=-1)[0]
confidence = np.max(predictions) * 100

print("\n--- INFERENCE RESULTS ---")
print(f"Predicted Class: {classes[predicted_class_idx]}")
print(f"Confidence:      {confidence:.2f}%")
print("-------------------------")
