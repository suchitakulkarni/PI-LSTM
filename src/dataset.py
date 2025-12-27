# utils.py (Complete, Final Version)

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import matplotlib.pyplot as plt 
import os


def simulate_harmonic_oscillator(timesteps=1000, dt=0.01, omega=2.0, noise_std=0.0):
    """Simulates a simple harmonic oscillator with optional measurement noise."""
    x = np.zeros(timesteps)
    v = np.zeros(timesteps)
    x[0] = 1.0
    for t in range(1, timesteps):
        a = -omega**2 * x[t-1]
        v[t] = v[t-1] + a * dt
        x[t] = x[t-1] + v[t] * dt

    # Add Gaussian measurement noise
    if noise_std > 0:
        np.random.seed(123)  # Different seed from anomaly injection
        x += np.random.normal(0, noise_std, timesteps)

    return x

def inject_perturbations(x, num_anomalies=10, severity=2.0):
    """Injects random Gaussian noise anomalies into the signal."""
    x_anomalous = x.copy()
    np.random.seed(42) 
    anomaly_indices = np.random.choice(len(x), num_anomalies, replace=False)
    for idx in anomaly_indices:
        if idx < len(x) - 1:
            x_anomalous[idx] += severity * np.random.randn()
    return x_anomalous, anomaly_indices

def create_rolling_windows(data, window_size):
    """Converts a 1D time series into a 2D array of overlapping sequences."""
    X = []
    for i in range(len(data) - window_size + 1):
        X.append(data[i:i+window_size])
    return np.array(X)
    
def prepare_data(data, window_size, scaler=None):
    """Scales the data and creates rolling windows."""
    data_reshaped = data.reshape(-1, 1) 
    
    if scaler is None:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_reshaped).flatten()
    else:
        scaled_data = scaler.transform(data_reshaped).flatten()
        
    windows = create_rolling_windows(scaled_data, window_size)
    
    # Reshape for LSTM input: (N, seq_len, 1)
    windows_tensor = windows[:, :, np.newaxis] 
    
    return windows_tensor, scaler, scaled_data