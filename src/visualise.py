# utils.py (Complete, Final Version)

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import matplotlib.pyplot as plt 
import os

def _get_full_reconstruction(reconstruction_windows, N_signal, window_size):
    """Helper to convert windowed reconstructions to a single signal."""
    N_windows = len(reconstruction_windows)
    print(f"[DEBUG] _get_full_reconstruction called with N_windows={len(reconstruction_windows)}, N_signal={N_signal}, window_size={window_size}")
    # Use the last point of each window and place it at the correct position
    reconstruction_full_pts = reconstruction_windows[:, window_size - 1, 0]
    
    reconstruction_padded = np.full(N_signal, np.nan)
    
    # Window i covers timesteps [i, i+window_size-1]
    # So the last point of window i corresponds to timestep i+window_size-1
    for i in range(N_windows):
        original_idx = i + window_size - 1
        if original_idx < N_signal:
            reconstruction_padded[original_idx] = reconstruction_full_pts[i]
    
    return reconstruction_padded

# --- Matplotlib Plotting Functions ---

def plot_physics_comparison_results(x_anomalous, errors_pinn, errors_standard, threshold_pinn, threshold_standard, window_size, anomaly_idxs, pinn_weight, filename="physics_anomaly_score_comparison.png"):
    """
    Compares the Anomaly Score (Reconstruction Error) of the PI-LSTM model
    vs. the Standard LSTM model using Matplotlib.
    """
    N_signal = len(x_anomalous)
    N_windows = len(errors_pinn)
    padding_length = window_size - 1
    time_steps = np.arange(N_signal)

    def pad_errors(errors):
        errors_padded = np.full(N_signal, np.nan)
        errors_padded[padding_length:padding_length + N_windows] = errors
        return errors_padded

    errors_pinn_padded = pad_errors(errors_pinn)
    errors_standard_padded = pad_errors(errors_standard)
    
    detected_idxs_pinn = np.where(errors_pinn_padded > threshold_pinn)[0]
    detected_idxs_standard = np.where(errors_standard_padded > threshold_standard)[0]

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(14, 10), sharex=True)
    
    # Subplot 1: Anomalous Signal
    ax1 = axes[0]
    ax1.plot(time_steps, x_anomalous, label='Anomalous Signal (Scaled)', color='#1F77B4', linewidth=1)
    if anomaly_idxs is not None:
        ax1.plot(anomaly_idxs, x_anomalous[anomaly_idxs], 'rx', markersize=8, label='True Anomalies', markeredgewidth=2)
    ax1.set_title('Anomalous Signal and True Anomaly Locations')
    ax1.set_ylabel('Amplitude (Scaled)')
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # Subplot 2: PI-LSTM Anomaly Score
    ax2 = axes[1]
    ax2.plot(time_steps, errors_pinn_padded, label='PI-LSTM Anomaly Score', color='#2ECC71', linewidth=1)
    ax2.axhline(threshold_pinn, color='#2ECC71', linestyle='--', linewidth=2, label=f'PI-LSTM Threshold ({threshold_pinn:.4f})')
    ax2.plot(detected_idxs_pinn, errors_pinn_padded[detected_idxs_pinn], 'go', markersize=4, alpha=0.7)
    ax2.set_title(f'Anomaly Score: Physics-Informed Autoencoder (Weight={pinn_weight:.2f})')
    ax2.set_ylabel('Error Magnitude (MSE)')
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle=':', alpha=0.6)

    # Subplot 3: Standard LSTM Anomaly Score
    ax3 = axes[2]
    ax3.plot(time_steps, errors_standard_padded, label='Standard LSTM Anomaly Score', color='#E74C3C', linewidth=1)
    ax3.axhline(threshold_standard, color='#E74C3C', linestyle='--', linewidth=2, label=f'Standard LSTM Threshold ({threshold_standard:.4f})')
    ax3.plot(detected_idxs_standard, errors_standard_padded[detected_idxs_standard], 'ro', markersize=4, alpha=0.7)
    ax3.set_title('Anomaly Score: Standard LSTM (Weight=0.0)')
    ax3.set_xlabel('Time Step Index')
    ax3.set_ylabel('Error Magnitude (MSE)')
    ax3.legend(loc='upper right')
    ax3.grid(True, linestyle=':', alpha=0.6)
    
    plt.suptitle("Comparison of Anomaly Scores (PI-LSTM vs. Standard LSTM)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(filename)
    plt.close()
    print(f"Plot saved to {filename}")


def plot_reconstruction_comparison(x_anomalous, recon_pinn_windows, recon_standard_windows, window_size, anomaly_idxs, filename="reconstruction_comparison.png"):
    """
    Plots the original signal and the reconstructed signals from both models.
    """
    N_signal = len(x_anomalous)
    time_steps = np.arange(N_signal)

    # Get full signal reconstructions
    recon_pinn_full = _get_full_reconstruction(recon_pinn_windows, N_signal, window_size)
    recon_standard_full = _get_full_reconstruction(recon_standard_windows, N_signal, window_size)

    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Original Signal
    ax.plot(time_steps, x_anomalous, label='Anomalous Signal (Scaled)', color='gray', linewidth=1.5, alpha=0.5)
    
    # PI-LSTM Reconstruction
    ax.plot(time_steps, recon_pinn_full, label='PI-LSTM Reconstruction', color='b', linestyle='-', linewidth=2, alpha=0.9)
    
    # Standard LSTM Reconstruction
    ax.plot(time_steps, recon_standard_full, label='Standard LSTM Reconstruction', color='#E74C3C', linestyle='--', linewidth=1.5, alpha=0.8)
    
    # Highlight true anomalies
    if anomaly_idxs is not None:
        ax.plot(anomaly_idxs, x_anomalous[anomaly_idxs], 'rx', markersize=8, label='True Anomalies', markeredgewidth=2)
        
    ax.set_title('Reconstruction Comparison: PI-LSTM vs. Standard LSTM')
    ax.set_xlabel('Time Step Index', fontsize=14)
    ax.set_ylabel('Amplitude (Scaled)', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Plot saved to {filename}")



def tune_threshold_f1(errors, true_anomaly_idxs, window_size, num_thresholds=100):
    thresholds = np.linspace(np.min(errors), np.max(errors), num_thresholds)
    results = []
    best_threshold = None
    best_f1 = -1
    best_metrics = {}

    window_centers = np.arange(window_size // 2, len(errors) + window_size // 2)

    for t in thresholds:
        preds = errors > t
        detected_idxs = window_centers[preds]

        # Match detected to true using proximity
        true_positives = [
            idx for idx in true_anomaly_idxs
            if any(abs(idx - d) <= window_size // 2 for d in detected_idxs)
        ]

        false_positives = [
            d for d in detected_idxs
            if not any(abs(d - idx) <= window_size // 2 for idx in true_anomaly_idxs)
        ]

        tp = len(true_positives)
        fp = len(false_positives)
        fn = len(true_anomaly_idxs) - tp

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        results.append({
            "threshold": t,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "detected_idxs": detected_idxs,
            "matched_true_idxs": true_positives # <--- NEW LINE
        })

        best = max(results, key=lambda x: x["f1"])

    return best, results

def plot_history(history, save_path="results/loss_curves.png"):

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(history['mse_loss'], label='MSE Loss')
    ax.plot(history['physics_loss'], label='Weighted Physics Loss')
    ax.plot(history['total_loss'], label='Total Loss')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Curves')
    ax.legend()
    ax.grid(True)
    ax.set_yscale('log')
    fig.tight_layout()
    fig.savefig(save_path)

    return fig


def get_detected_true_anomalies(anomaly_idxs, model_detections, window_size):
    """
    Checks which true anomaly indices were covered by the model's detections.

    Args:
        anomaly_idxs (np.ndarray): Array of time indices where true anomalies exist (Red 'X's).
        model_detections (np.ndarray): Array of time indices of detected anomaly windows (e.g., Green Circles).
        window_size (int): The size of the sliding window used by the autoencoder.

    Returns:
        tuple: (detected_anomaly_idxs, missed_anomaly_idxs)
    """
    detected_idxs = []
    missed_idxs = []
    
    # 1. Group the anomaly_idxs into unique 'events' (clusters of adjacent 'X's)
    # Since anomaly_idxs are usually time steps, we can define an event as a cluster.
    # For simplicity, we'll treat each unique index in anomaly_idxs as a potential event
    # and check if it was covered.
    
    # In a typical time-series setting, 'True Anomalies' are defined by a *range*
    # (e.g., a window of 30 steps). For this code, we assume the true anomaly
    # is covered if a detection occurs within a small proximity (defined by window_size).
    
    # We will use a simple mapping: a true anomaly at index 'i' is detected if
    # the model flags *any* time step 'j' that is near 'i'.
    # Since the anomaly score is computed for a *window*, we check for simple overlap.
    
    # Define a set of all time steps flagged by the model for quick lookup
    model_detection_set = set(model_detections)
    
    # Iterate over the true anomaly indices
    for idx in anomaly_idxs:
        # Check a small window around the true anomaly index
        # A true anomaly at idx is detected if any of the model_detections is within 
        # the window corresponding to the anomaly score
        is_detected = False
        
        # We need to consider the window_size logic. If a model detects a window
        # centered at 'c', it covers the range [c - W/2, c + W/2].
        # A simple check: if the true anomaly index 'idx' is within the range
        # covered by a detected window, it's a TP.
        
        # Given how 'model_detections' are calculated (window centers), let's check
        # if the true anomaly index 'idx' is in the set of all time steps covered 
        # by the detection windows.
        
        # The easiest approach is to check if any model detection is "close enough"
        # to the true anomaly index. Let's use the window size as a tolerance.
        
        # A more robust check: An anomaly at 'idx' is detected if any model_detection 
        # index 'd' is close to 'idx'.
        if np.any(np.abs(model_detections - idx) <= window_size // 2):
            is_detected = True
        
        if is_detected:
            detected_idxs.append(idx)
        else:
            missed_idxs.append(idx)
            
    return np.array(detected_idxs), np.array(missed_idxs)


def plot_detected_anomalies_comparison(x_anomalous, errors_pinn, errors_standard, 
                                       threshold_pinn, threshold_standard, 
                                       window_size, anomaly_idxs, 
                                       filename="results/detected_anomalies_comparison_TP.png"):
    """
    Plots the anomalous signal with detected anomalies overlaid from both models,
    including specific markers for True Positives (TPs).
    """
    N_signal = len(x_anomalous)
    time_steps = np.arange(N_signal)
    
    # Calculate window centers
    # Note: len(errors) is N_signal - window_size + 1 (the number of windows)
    window_centers = np.arange(window_size // 2, len(errors_pinn) + window_size // 2)
    
    # Get all window centers that were flagged as anomalous
    pinn_detections_centers = window_centers[errors_pinn > threshold_pinn]
    standard_detections_centers = window_centers[errors_standard > threshold_standard]
    
    # ------------------------------------------------------------------
    # Identify which True Anomalies ('X's) were detected (TP)
    # ------------------------------------------------------------------
    
    # PI-LSTM TP
    pinn_tp_idxs, _ = get_detected_true_anomalies(
        anomaly_idxs, pinn_detections_centers, window_size
    )
    # Standard LSTM TP
    standard_tp_idxs, _ = get_detected_true_anomalies(
        anomaly_idxs, standard_detections_centers, window_size
    )

    # Convert to sets to find unique TPs (where one model caught it and the other didn't)
    pinn_tp_set = set(pinn_tp_idxs)
    standard_tp_set = set(standard_tp_idxs)
    
    # Anomalies detected ONLY by PI-LSTM (The one extra TP)
    pinn_only_tp_idxs = np.array(list(pinn_tp_set - standard_tp_set))
    # Anomalies detected ONLY by Standard LSTM (If there were any)
    standard_only_tp_idxs = np.array(list(standard_tp_set - pinn_tp_set))
    # Anomalies detected by BOTH
    both_tp_idxs = np.array(list(pinn_tp_set.intersection(standard_tp_set)))
    
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot anomalous signal
    ax.plot(time_steps, x_anomalous, label='Anomalous Signal (Scaled)', 
            color='#1F77B4', linewidth=1.5, alpha=0.8)
    
    # 1. Plot all True Anomalies (Missed + Detected) as the base 'X'
    if anomaly_idxs is not None:
        ax.scatter(anomaly_idxs, x_anomalous[anomaly_idxs], 
                  color='red', s=100, marker='X', 
                  label='True Anomalies (Ground Truth)', zorder=5, edgecolors='black', linewidths=1.5)
    
    # 2. Plot PI-LSTM detections (Green Circles)
    pinn_y_vals = x_anomalous[pinn_detections_centers]
    ax.scatter(pinn_detections_centers, pinn_y_vals, 
              color='#2ECC71', s=80, marker='o', alpha=0.6,
              label=f'PI-LSTM Detections (FP + TP Sprawl, n={len(pinn_detections_centers)})', 
              zorder=4, edgecolors='darkgreen', linewidths=1)
    
    # 3. Plot Standard LSTM detections (Red Squares)
    standard_y_vals = x_anomalous[standard_detections_centers]
    ax.scatter(standard_detections_centers, standard_y_vals, 
              color='#E74C3C', s=60, marker='s', alpha=0.6,
              label=f'Standard LSTM Detections (FP + TP Sprawl, n={len(standard_detections_centers)})', 
              zorder=3, edgecolors='darkred', linewidths=1)
              
    # ------------------------------------------------------------------
    # Plot True Positives (TP) markers on top of the 'X's
    # ------------------------------------------------------------------
    
    # Anomalies detected ONLY by PI-LSTM (The one that accounts for 28 vs 27)
    if len(pinn_only_tp_idxs) > 0:
        ax.scatter(pinn_only_tp_idxs, x_anomalous[pinn_only_tp_idxs], 
                   color='darkgreen', s=200, marker='*', 
                   label=f'TP - PI-LSTM ONLY (n={len(pinn_only_tp_idxs)})', 
                   zorder=6, edgecolors='white', linewidths=1)

    # Anomalies detected ONLY by Standard LSTM
    if len(standard_only_tp_idxs) > 0:
        ax.scatter(standard_only_tp_idxs, x_anomalous[standard_only_tp_idxs], 
                   color='darkred', s=200, marker='^', 
                   label=f'TP - Standard LSTM ONLY (n={len(standard_only_tp_idxs)})', 
                   zorder=6, edgecolors='white', linewidths=1)

    # Anomalies detected by BOTH (Majority)
    if len(both_tp_idxs) > 0:
        ax.scatter(both_tp_idxs, x_anomalous[both_tp_idxs], 
                   color='purple', s=200, marker='P', 
                   label=f'TP - BOTH Models (n={len(both_tp_idxs)})', 
                   zorder=6, edgecolors='white', linewidths=1)


    # Update the title and labels for clarity
    ax.set_title('Detected Anomalies: PI-LSTM vs. Standard LSTM (with TP Highlighting)', fontsize=14)
    ax.set_xlabel('Time Step Index', fontsize=14)
    ax.set_ylabel('Amplitude (Scaled)', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)

    ax.tick_params(axis='both', which='major', labelsize=12)
    #ax.tick_params(axis='both', which='minor', labelsize=8)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Plot saved to {filename}")

def analyze_phase_shift(model, test_loader, window_size, device, model_name="Model"):
    """
    Analyzes phase shift in model reconstructions.
    """
    model.eval()
    
    all_correlations = []
    all_lags = []
    
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            if batch_idx >= 10:  # Analyze first 10 batches
                break
                
            data_window = data[0].to(device)
            reconstruction = model(data_window)
            
            # Analyze each window in batch
            for i in range(min(5, data_window.shape[0])):  # First 5 windows per batch
                original = data_window[i, :, 0].cpu().numpy()
                recon = reconstruction[i, :, 0].cpu().numpy()
                
                # Compute cross-correlation to find lag
                correlation = np.correlate(original, recon, mode='full')
                lag = np.argmax(correlation) - (window_size - 1)
                max_corr = np.max(correlation) / (np.linalg.norm(original) * np.linalg.norm(recon))
                
                all_correlations.append(max_corr)
                all_lags.append(lag)
    
    avg_lag = np.mean(all_lags)
    std_lag = np.std(all_lags)
    avg_corr = np.mean(all_correlations)
    
    print(f"\n{model_name} Phase Shift Analysis:")
    print(f"  Average lag: {avg_lag:.2f} timesteps (std: {std_lag:.2f})")
    print(f"  Average correlation: {avg_corr:.4f}")
    print(f"  Lag distribution: min={np.min(all_lags)}, max={np.max(all_lags)}, median={np.median(all_lags)}")
    
    return {
        'avg_lag': avg_lag,
        'std_lag': std_lag,
        'avg_corr': avg_corr,
        'all_lags': all_lags,
        'all_correlations': all_correlations
    }

def plot_phase_shift_analysis(x_anomalous, recon_pinn_windows, recon_standard_windows, 
                              window_size, window_idx=100, 
                              filename="results/phase_shift_detail.png"):
    """
    Detailed plot showing phase shift for a single window.
    """
    original_window = x_anomalous[window_idx:window_idx+window_size]
    pinn_window = recon_pinn_windows[window_idx, :, 0]
    standard_window = recon_standard_windows[window_idx, :, 0]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Top left: Overlaid signals
    ax1 = axes[0, 0]
    timesteps = np.arange(window_size)
    ax1.plot(timesteps, original_window, 'k-', linewidth=2, label='Original', marker='o')
    ax1.plot(timesteps, pinn_window, 'g--', linewidth=2, label='PI-LSTM', marker='s', alpha=0.7)
    ax1.plot(timesteps, standard_window, 'r:', linewidth=2, label='Standard LSTM', marker='^', alpha=0.7)
    ax1.set_title(f'Window {window_idx}: Signal Comparison')
    ax1.set_xlabel('Timestep within window')
    ax1.set_ylabel('Amplitude')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Top right: Point-by-point errors
    ax2 = axes[0, 1]
    pinn_errors = np.abs(original_window - pinn_window)
    standard_errors = np.abs(original_window - standard_window)
    ax2.plot(timesteps, pinn_errors, 'g-', linewidth=2, label='PI-LSTM Error', marker='s')
    ax2.plot(timesteps, standard_errors, 'r-', linewidth=2, label='Standard LSTM Error', marker='^')
    ax2.set_title('Point-wise Absolute Error')
    ax2.set_xlabel('Timestep within window')
    ax2.set_ylabel('Absolute Error')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Bottom left: Cross-correlation for PINN
    ax3 = axes[1, 0]
    pinn_xcorr = np.correlate(original_window, pinn_window, mode='full')
    lags = np.arange(-window_size+1, window_size)
    ax3.plot(lags, pinn_xcorr, 'g-', linewidth=2)
    max_idx = np.argmax(pinn_xcorr)
    ax3.axvline(lags[max_idx], color='red', linestyle='--', label=f'Peak at lag={lags[max_idx]}')
    ax3.set_title('PI-LSTM Cross-Correlation')
    ax3.set_xlabel('Lag')
    ax3.set_ylabel('Correlation')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Bottom right: Cross-correlation for Standard LSTM
    ax4 = axes[1, 1]
    standard_xcorr = np.correlate(original_window, standard_window, mode='full')
    ax4.plot(lags, standard_xcorr, 'r-', linewidth=2)
    max_idx = np.argmax(standard_xcorr)
    ax4.axvline(lags[max_idx], color='red', linestyle='--', label=f'Peak at lag={lags[max_idx]}')
    ax4.set_title('Standard LSTM Cross-Correlation')
    ax4.set_xlabel('Lag')
    ax4.set_ylabel('Correlation')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Phase shift analysis plot saved to {filename}")

def inject_physics_violating_perturbations(x, num_anomalies=10, duration=5, omega=2.0, dt=0.01):
        """
        Injects anomalies that violate the harmonic oscillator equation.
        These should be easier for PI-LSTM to detect than standard LSTM.
        """
        x_anomalous = x.copy()
        np.random.seed(42)
        anomaly_indices = np.random.choice(len(x) - duration - 1, num_anomalies, replace=False)
        
        for start_idx in anomaly_indices:
            severity = np.random.random()

            # Create a short segment that doesn't follow d^2x/dt^2 = -omega^2 * x
            # For example: add a linear trend (which violates the oscillator equation)
            for i in range(duration):
                idx = start_idx + i
                if idx < len(x):
                    # Add linear drift that violates physics
                    x_anomalous[idx] += np.random.choice([-1,1])*severity * i / duration
        
        return x_anomalous, anomaly_indices

def plot_physics_violation_types(x_clean, x_anomalous, anomaly_dict, window_size, 
                                  filename="results/physics_violation_types.png"):
    """
    Visualizes different types of physics violations in the signal.
    
    Args:
        x_clean: Clean signal
        x_anomalous: Signal with anomalies
        anomaly_dict: Dictionary with keys 'frequency', 'phase', 'damping'
        window_size: Window size for highlighting
        filename: Save path
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    time_steps = np.arange(len(x_anomalous))
    
    # Plot 1: Clean signal
    axes[0].plot(time_steps, x_clean, 'b-', linewidth=1, alpha=0.7)
    axes[0].set_title('Original Clean Signal')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Anomalous signal with all violations marked
    axes[1].plot(time_steps, x_anomalous, 'k-', linewidth=1, alpha=0.7)
    
    if 'frequency' in anomaly_dict:
        for idx in anomaly_dict['frequency']:
            axes[1].axvspan(idx, idx + 20, alpha=0.3, color='red', label='Freq' if idx == anomaly_dict['frequency'][0] else '')
    
    if 'phase' in anomaly_dict:
        axes[1].scatter(anomaly_dict['phase'], x_anomalous[anomaly_dict['phase']], 
                       color='blue', marker='v', s=100, label='Phase Jump', zorder=5)
    
    if 'damping' in anomaly_dict:
        for idx in anomaly_dict['damping']:
            axes[1].axvspan(idx, idx + 30, alpha=0.3, color='green', label='Damp' if idx == anomaly_dict['damping'][0] else '')
    
    axes[1].set_title('Anomalous Signal with Physics Violations')
    axes[1].set_ylabel('Amplitude')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Difference (anomalous - clean)
    diff = x_anomalous - x_clean
    axes[2].plot(time_steps, diff, 'r-', linewidth=1)
    axes[2].set_title('Difference (Anomalous - Clean)')
    axes[2].set_ylabel('Difference')
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(0, color='k', linestyle='--', alpha=0.5)
    
    # Plot 4: Local frequency estimate (using zero-crossings)
    # Simple frequency estimation
    zero_crossings = np.where(np.diff(np.sign(x_anomalous)))[0]
    if len(zero_crossings) > 1:
        local_periods = np.diff(zero_crossings) * 2  # Full period = 2 * half period
        local_freq = 2 * np.pi / (local_periods * 0.01)  # Assuming dt=0.01
        axes[3].plot(zero_crossings[:-1], local_freq, 'go-', linewidth=1, markersize=3)
        axes[3].axhline(2.0, color='b', linestyle='--', linewidth=2, label='True frequency')
        axes[3].set_title('Estimated Local Frequency')
        axes[3].set_ylabel('Frequency (rad/s)')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
    
    axes[3].set_xlabel('Time Step')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Physics violation plot saved to {filename}")

def plot_overfitting_analysis(history, physics_weight, 
                               filename="results/overfitting_analysis.png"):
    """
    Comprehensive overfitting visualization.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    epochs = np.arange(1, len(history['train_total']) + 1)
    
    # Plot 1: Total Loss (Train vs Val)
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['train_total'], 'b-', linewidth=2, label='Train Total Loss')
    ax1.plot(epochs, history['val_total'], 'r-', linewidth=2, label='Val Total Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Total Loss')
    ax1.set_title('Total Loss: Train vs Validation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Highlight overfitting region (where val > train significantly)
    overfitting_start = None
    for i, gap in enumerate(history['overfitting_gap']):
        if gap > 0.1 * history['train_total'][i]:  # 10% threshold
            overfitting_start = i
            break
    if overfitting_start:
        ax1.axvline(overfitting_start + 1, color='orange', linestyle='--', 
                   linewidth=2, label=f'Overfitting starts ~{overfitting_start+1}')
        ax1.legend()
    
    # Plot 2: MSE Loss (Train vs Val)
    ax2 = axes[0, 1]
    ax2.plot(epochs, history['train_mse'], 'b-', linewidth=2, label='Train MSE')
    ax2.plot(epochs, history['val_mse'], 'r-', linewidth=2, label='Val MSE')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MSE Loss')
    ax2.set_title('Reconstruction Loss: Train vs Validation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # Plot 3: Physics Loss (Train vs Val)
    ax3 = axes[1, 0]
    if physics_weight > 0:
        ax3.plot(epochs, history['train_physics'], 'b-', linewidth=2, label='Train Physics')
        ax3.plot(epochs, history['val_physics'], 'r-', linewidth=2, label='Val Physics')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Physics Loss')
        ax3.set_title(f'Physics Loss (Weight={physics_weight:.3f}): Train vs Validation')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
    else:
        ax3.text(0.5, 0.5, 'Physics Weight = 0\n(Standard Autoencoder)', 
                ha='center', va='center', fontsize=14, transform=ax3.transAxes)
        ax3.set_title('Physics Loss (Not Used)')
    
    # Plot 4: Overfitting Gap
    ax4 = axes[1, 1]
    ax4.plot(epochs, history['overfitting_gap'], 'purple', linewidth=2, label='Val - Train')
    ax4.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax4.fill_between(epochs, 0, history['overfitting_gap'], 
                     where=(np.array(history['overfitting_gap']) > 0),
                     alpha=0.3, color='red', label='Overfitting Region')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss Gap (Val - Train)')
    ax4.set_title('Overfitting Gap Analysis')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'Overfitting Analysis (Physics Weight = {physics_weight:.3f})', 
                 fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Overfitting analysis saved to {filename}")


def plot_generalization_metrics(history_pinn, history_standard, 
                                 filename="results/generalization_comparison.png"):
    """
    Compare generalization between PINN and Standard AE.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs_pinn = np.arange(1, len(history_pinn['train_total']) + 1)
    epochs_std = np.arange(1, len(history_standard['train_total']) + 1)
    
    # Plot 1: Overfitting Gap Comparison
    ax1 = axes[0]
    ax1.plot(epochs_pinn, history_pinn['overfitting_gap'], 
            'g-', linewidth=2, label='PINN')
    ax1.plot(epochs_std, history_standard['overfitting_gap'], 
            'r-', linewidth=2, label='Standard AE')
    ax1.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Overfitting Gap (Val - Train)')
    ax1.set_title('Generalization: PINN vs Standard AE')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Final Gap Comparison (Bar Chart)
    ax2 = axes[1]
    final_gap_pinn = history_pinn['overfitting_gap'][-1]
    final_gap_std = history_standard['overfitting_gap'][-1]
    
    bars = ax2.bar(['PINN', 'Standard AE'], 
                   [final_gap_pinn, final_gap_std],
                   color=['green', 'red'], alpha=0.7)
    ax2.axhline(0, color='k', linestyle='--', linewidth=1)
    ax2.set_ylabel('Final Overfitting Gap')
    ax2.set_title('Final Generalization Performance')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.5f}',
                ha='center', va='bottom' if height > 0 else 'top')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Generalization comparison saved to {filename}")