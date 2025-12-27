# main_comparison.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import sys

# Ensure imports from sibling files work
#sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import Config
from src.model import LSTMAutoencoder
from src.train import train_model
from src.dataset import (simulate_harmonic_oscillator,
    inject_perturbations,
    prepare_data)
from src.evaluate import evaluate_model    
from src.visualise import (
    plot_physics_comparison_results, 
    plot_reconstruction_comparison,
    tune_threshold_f1, 
    plot_history,
    plot_detected_anomalies_comparison,
    plot_phase_shift_analysis,
    analyze_phase_shift
)

def run_single_model(config, X_train_tensor, train_loader, test_loader, physics_loss_weight, device, run_name, anomaly_idxs):
    """
    Runs a single training and evaluation pipeline.
    Returns: (errors_np, threshold, reconstruction_windows, detected_windows_count, detected_anomaly_tp)
    """
    print(f"\n--- Running: {run_name} (Physics Weight: {physics_loss_weight:.2f}) ---")
    
    model = LSTMAutoencoder(
        seq_len=config.WINDOW_SIZE, 
        hidden_dim=config.HIDDEN_DIM
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.MSELoss(reduction='none') 
    
    # Train the model
    model, history = train_model(
        model=model, 
        train_loader=train_loader, 
        criterion=criterion, 
        optimizer=optimizer, 
        num_epochs=config.NUM_EPOCHS,
        physics_loss_weight=physics_loss_weight, 
        dt=config.DT, 
        device=device
    )
    if physics_loss_weight > 0:
        plot_history(history, save_path = os.path.join(config.RESULTS_DIR, 'physics_loss.png'))
    else: plot_history(history, save_path = os.path.join(config.RESULTS_DIR, 'mse_loss.png'))

    # Evaluate the model
    errors_np, reconstructed_windows = evaluate_model(
        model=model, 
        data_loader=test_loader, 
        criterion=nn.MSELoss(reduction='none'), 
        device=device
    )
    # After PINN evaluation
    #print("\n--- Analyzing PINN Phase Shift ---")
    pinn_phase_info = analyze_phase_shift(model, test_loader, config.WINDOW_SIZE, device, "PINN")
    # --- Anomaly Detection and Metrics (F1-Optimized) ---
    
    # Use the F1-optimized threshold to calculate True Positives (TP)
    best_result, _ = tune_threshold_f1(errors_np, anomaly_idxs, config.WINDOW_SIZE)
    threshold = best_result["threshold"]
    
    # Counts based on the optimal threshold
    detected_windows_count = len(best_result["detected_idxs"])
    detected_anomaly_tp = best_result["tp"]
    
    # Print the relevant detection stats
    print(f"Evaluation: F1-optimized threshold: {threshold:.4f}")
    print(f"Evaluation: Windows flagged as anomalous: {detected_windows_count}")
    print(f"Evaluation: Unique anomalies detected (True Positives): {detected_anomaly_tp}")
    
    # Return the new metric: detected_anomaly_tp
    return errors_np, threshold, reconstructed_windows, detected_windows_count, detected_anomaly_tp


def run_comparison(config: Config):
    """
    Runs two pipelines (PINN and Standard AE) for comparison, prints results, and plots.
    """
    print("--- 1. Configuration Loaded and Data Prepared ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # For training data (no noise)
    x_clean = simulate_harmonic_oscillator(
    timesteps=int(config.TIMESTEPS/3),
    dt=config.DT,
    omega=config.OMEGA,
    noise_std=0.0  
    )

    # For test data baseline (with noise)
    x_with_noise = simulate_harmonic_oscillator(
    timesteps=config.TIMESTEPS,
    dt=config.DT,
    omega=config.OMEGA,
    noise_std=0.5
    )

    X_train_np, scaler, _ = prepare_data(x_clean, config.WINDOW_SIZE)
    X_train_tensor = torch.from_numpy(X_train_np).float().to(device)
    train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    # Anomalous data for testing (Shared by both models)
    x_anomalous, anomaly_idxs = inject_perturbations(
        x_with_noise, 
        num_anomalies=config.NUM_ANOMALIES, 
        severity=config.SEVERITY
    )

    X_test_anom_np, _, scaled_x_anomalous = prepare_data(
        x_anomalous, 
        config.WINDOW_SIZE, 
        scaler=scaler
    )
    X_test_anom_tensor = torch.from_numpy(X_test_anom_np).float().to(device)
    test_dataset = TensorDataset(X_test_anom_tensor, X_test_anom_tensor)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # --- Run 1: PINN-Enabled Autoencoder ---
    pinn_weight = config.PHYSICS_LOSS_WEIGHT
    errors_pinn, threshold_pinn, recon_pinn_windows, detected_pinn_windows, detected_pinn_anomalies = run_single_model(
        config, X_train_tensor, train_loader, test_loader, pinn_weight, device, "PINN Autoencoder", anomaly_idxs
    )

    
    # --- Run 2: Standard Autoencoder ---
    standard_ae_weight = 0.0
    errors_standard, threshold_standard, recon_standard_windows, detected_standard_windows, detected_standard_anomalies = run_single_model(
        config, X_train_tensor, train_loader, test_loader, standard_ae_weight, device, "Standard Autoencoder", anomaly_idxs
    )

    # --- Print Comparison Summary (Updated with TP count) ---
    print("\n" + "="*70)
    print("           ANOMALY DETECTION COMPARISON (F1-OPTIMIZED THRESHOLD)")
    print("="*70)
    print(f"Total Injected Anomalies (Ground Truth): {config.NUM_ANOMALIES}")
    print("\n| Model                  | Unique Anomalies Detected (TP) | Total Windows Flagged | F1-Threshold |")
    print("|:-----------------------|:------------------------------:|:---------------------:|:------------:|")
    print(f"| PINN (Weight={pinn_weight:.2f})      | {detected_pinn_anomalies:30} | {detected_pinn_windows:21} | {threshold_pinn:12.4f} |")
    print(f"| Standard AE (Weight=0.00)| {detected_standard_anomalies:30} | {detected_standard_windows:21} | {threshold_standard:12.4f} |")
    print("="*70 + "\n")


    # --- Generate Comparison Plots ---
    
    # Plot 1: Anomaly Scores
    print("--- Generating Anomaly Score Comparison Plot (physics_anomaly_score_comparison.png) ---")
    # Note: Using F1-optimized thresholds for the plot
    plot_physics_comparison_results(
        x_anomalous=scaled_x_anomalous,
        errors_pinn=errors_pinn,
        errors_standard=errors_standard,
        threshold_pinn=threshold_pinn,
        threshold_standard=threshold_standard,
        window_size=config.WINDOW_SIZE,
        anomaly_idxs=anomaly_idxs,
        pinn_weight=pinn_weight,
        filename=os.path.join(config.RESULTS_DIR, "physics_anomaly_score_comparison.png")
    )
    
    # Plot 2: Reconstruction Comparison
    print("\n--- Generating Reconstruction Comparison Plot (reconstruction_comparison.png) ---")
    plot_reconstruction_comparison(
        x_anomalous=scaled_x_anomalous,
        recon_pinn_windows=recon_pinn_windows,
        recon_standard_windows=recon_standard_windows,
        window_size=config.WINDOW_SIZE,
        anomaly_idxs=anomaly_idxs,
        filename=os.path.join(config.RESULTS_DIR, "reconstruction_comparison.png")
    )
    # Plot 3: Detected Anomalies Overlay
    print("\n--- Generating Detected Anomalies Comparison Plot (detected_anomalies_comparison.png) ---")
    plot_detected_anomalies_comparison(
        x_anomalous=scaled_x_anomalous,
        errors_pinn=errors_pinn,
        errors_standard=errors_standard,
        threshold_pinn=threshold_pinn,
        threshold_standard=threshold_standard,
        window_size=config.WINDOW_SIZE,
        anomaly_idxs=anomaly_idxs,
        filename=os.path.join(config.RESULTS_DIR, "detected_anomalies_comparison.png")
    )
    
    print("\n--- Comparison Pipeline Finished Successfully ---")
    

if __name__ == "__main__":
    # NOTE: You must have a config.py file with the necessary Config class
    app_config = Config() 
    run_comparison(app_config)
