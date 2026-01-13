# utils.py (Complete, Final Version)

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
import torch.nn as nn
import matplotlib.pyplot as plt 
import os

# dataset.py

def simulate_harmonic_oscillator(timesteps=1000, dt=0.01, omega=2.0, noise_std=0.0, seed=None):
    """Simulates a simple harmonic oscillator with optional measurement noise."""
    if seed is not None:
        np.random.seed(seed)
    
    x = np.zeros(timesteps)
    v = np.zeros(timesteps)
    x[0] = 1.0
    for t in range(1, timesteps):
        a = -omega**2 * x[t-1]
        v[t] = v[t-1] + a * dt
        x[t] = x[t-1] + v[t] * dt

    # Add Gaussian measurement noise
    if noise_std > 0:
        x += np.random.normal(0, noise_std, timesteps)

    return x

def inject_perturbations(x, num_anomalies=10, severity=2.0, seed=None):
    """Injects random Gaussian noise anomalies into the signal."""
    if seed is not None:
        np.random.seed(seed)
    
    x_anomalous = x.copy()
    anomaly_indices = np.random.choice(len(x), num_anomalies, replace=False)
    for idx in anomaly_indices:
        if idx < len(x) - 1:
            x_anomalous[idx] += severity * np.random.randn()
    return x_anomalous, anomaly_indices

def inject_frequency_violations(x, num_anomalies=10, duration=20, omega_base=2.0, dt=0.01, seed=None):
    """
    Injects frequency-violating anomalies by locally changing oscillation frequency.
    """
    if seed is not None:
        np.random.seed(seed)
    
    x_anomalous = x.copy()
    
    max_start = len(x) - duration - 1
    if max_start <= 0:
        return x_anomalous, np.array([])
    
    anomaly_indices = np.random.choice(max_start, num_anomalies, replace=False)
    
    for start_idx in anomaly_indices:
        omega_anomalous = omega_base * np.random.uniform(0.5, 2.0)
        A = x_anomalous[start_idx]
        t_segment = np.arange(duration) * dt
        phase_offset = np.arccos(A / (np.abs(A) + 1e-8))
        x_anomalous[start_idx:start_idx + duration] = (
            A * np.cos(omega_anomalous * t_segment + phase_offset)
        )
    
    return x_anomalous, anomaly_indices

def inject_phase_discontinuities(x, num_anomalies=10, phase_shift_range=(np.pi/4, np.pi), seed=None):
    """
    Injects sudden phase jumps that violate smooth oscillation.
    """
    if seed is not None:
        np.random.seed(seed)
    
    x_anomalous = x.copy()
    anomaly_indices = np.random.choice(len(x) - 1, num_anomalies, replace=False)
    anomaly_indices = np.sort(anomaly_indices)
    
    for idx in anomaly_indices:
        phase_shift = np.random.uniform(*phase_shift_range)
        current_val = x_anomalous[idx]
        next_val = x_anomalous[idx + 1]
        x_anomalous[idx + 1] = current_val * np.cos(phase_shift) - next_val * np.sin(phase_shift)
    
    return x_anomalous, anomaly_indices

def inject_damping_violations(x, num_anomalies=10, duration=30, damping_factor=0.95, dt=0.01, seed=None):
    """
    Injects artificial damping or amplification that violates energy conservation.
    """
    if seed is not None:
        np.random.seed(seed)
    
    x_anomalous = x.copy()
    
    max_start = len(x) - duration - 1
    if max_start <= 0:
        return x_anomalous, np.array([])
    
    anomaly_indices = np.random.choice(max_start, num_anomalies, replace=False)
    
    for start_idx in anomaly_indices:
        factor = np.random.choice([
            np.random.uniform(0.85, 0.98),
            np.random.uniform(1.02, 1.15)
        ])
        envelope = factor ** np.arange(duration)
        x_anomalous[start_idx:start_idx + duration] *= envelope
    
    return x_anomalous, anomaly_indices

def inject_combined_physics_violations(x, num_anomalies=10, omega_base=2.0, dt=0.01, seed=None):
    """
    Combines multiple types of physics violations for comprehensive testing.
    """
    if seed is not None:
        np.random.seed(seed)
    
    x_anomalous = x.copy()
    
    n_freq = num_anomalies // 3
    n_phase = num_anomalies // 3
    n_damp = num_anomalies - n_freq - n_phase
    
    # Use different sub-seeds for each type
    x_anomalous, freq_idxs = inject_frequency_violations(
        x_anomalous, n_freq, duration=20, omega_base=omega_base, dt=dt, seed=seed
    )
    
    x_anomalous, phase_idxs = inject_phase_discontinuities(
        x_anomalous, n_phase, phase_shift_range=(np.pi/4, np.pi), seed=seed+1 if seed else None
    )
    
    x_anomalous, damp_idxs = inject_damping_violations(
        x_anomalous, n_damp, duration=30, damping_factor=0.95, dt=dt, seed=seed+2 if seed else None
    )
    
    anomaly_dict = {
        'frequency': freq_idxs,
        'phase': phase_idxs,
        'damping': damp_idxs,
        'all': np.concatenate([freq_idxs, phase_idxs, damp_idxs])
    }
    
    return x_anomalous, anomaly_dict

def inject_phase_frequency_anomalies(x, anomaly_idxs, dt, freq_shift=0.1, phase_shift=np.pi/8, seed=None):
    """
    Injects subtle anomalies by modifying phase or frequency in specific segments.
    """
    if seed is not None:
        np.random.seed(seed)
    
    x_anom = x.copy()
    for idx in anomaly_idxs:
        segment_len = min(50, len(x) - idx)
        t = np.arange(segment_len) * dt
        x_anom[idx:idx+segment_len] = x_anom[idx:idx+segment_len] * np.cos(freq_shift * t + phase_shift)
    return x_anom


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
        #scaler = MinMaxScaler()
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_reshaped).flatten()
    else:
        scaled_data = scaler.transform(data_reshaped).flatten()
        
    windows = create_rolling_windows(scaled_data, window_size)
    
    # Reshape for LSTM input: (N, seq_len, 1)
    windows_tensor = windows[:, :, np.newaxis] 
    
    return windows_tensor, scaler, scaled_data