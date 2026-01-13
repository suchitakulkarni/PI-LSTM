# config.py
import os
class Config:
    """
    Configuration settings for the Harmonic Oscillator Anomaly Detection Model.
    """
    # --- Reproducibility Settings ---
    RANDOM_STATE = 42         # Master seed for reproducibility
    
    # --- Data Simulation Settings ---
    TIMESTEPS = 1000          
    DT = 0.01                 
    OMEGA = 2.0               # Oscillation frequency
    NUM_ANOMALIES = 30        # Number of anomalies original 20
    SEVERITY = 2              # Perturbation strength original 0.5

    # --- Data Preparation Settings ---
    WINDOW_SIZE = 5           # Length of the rolling window sequence original is 20
    BATCH_SIZE = 64           
    TEST_SIZE = 0.3           
    # RANDOM_STATE already defined above

    # --- Model Hyperparameters ---
    HIDDEN_DIM = 256          # Hidden units in the LSTM original was 64
    NUM_EPOCHS = 1000         # original was 50
    LEARNING_RATE = 1e-4      # original was 1e-3

    # --- Loss Function Settings ---
    RECONSTRUCTION_LOSS_WEIGHT = 1.0 
    PHYSICS_LOSS_WEIGHT = 0.2 # Weight for the physics-informed loss original was 0.01, last working was 0.1
    
    RESULTS_DIR = 'results'
    os.makedirs(RESULTS_DIR, exist_ok = True)