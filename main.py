# main_comparison.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import sys
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Ensure imports from sibling files work
#sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import Config
from src.model import LSTMAutoencoder
from src.train import (train_model, compute_residual)
from src.dataset import (simulate_harmonic_oscillator,
    inject_perturbations,
    prepare_data, inject_combined_physics_violations, 
    inject_damping_violations, inject_phase_discontinuities,inject_frequency_violations)
from src.evaluate import evaluate_model    
from src.visualise import (
    plot_physics_comparison_results, 
    plot_reconstruction_comparison,
    tune_threshold_f1, 
    plot_history,
    plot_detected_anomalies_comparison,
    plot_phase_shift_analysis,
    analyze_phase_shift, 
    plot_physics_violation_types
)
from src.utils import (set_all_seeds, seed_worker)

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.train import calculate_physics_loss

def unit_leakage_test_suite(dt, omega_true, amplitudes=[0.5, 1.0, 2.0], scalers=["StandardScaler", "MinMaxScaler"], batch_size=5):
    """
    Comprehensive check for unit leakage in physics loss.
    - dt: time step
    - omega_true: physical frequency
    - amplitudes: list of trajectory amplitudes to test
    - scalers: list of scalers to test
    - batch_size: number of trajectories in a batch
    """
    T = 5.0
    t = np.arange(0, T, dt)

    for scaler_type in scalers:
        # Initialize scaler
        if scaler_type == "StandardScaler":
            from sklearn.preprocessing import StandardScaler
            scaler_class = StandardScaler
        elif scaler_type == "MinMaxScaler":
            from sklearn.preprocessing import MinMaxScaler
            scaler_class = MinMaxScaler
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")

        for A in amplitudes:
            # Create batched trajectories with different phases
            batch = np.zeros((batch_size, len(t)))
            for i in range(batch_size):
                phase = i * np.pi / 4  # different phase for each trajectory
                batch[i] = A * np.cos(omega_true * t + phase)

            # Scale each trajectory
            scaled_batch = []
            scalers_used = []
            for traj in batch:
                scaler = scaler_class()
                scaled_traj = scaler.fit_transform(traj.reshape(-1, 1)).flatten()
                scaled_batch.append(scaled_traj)
                scalers_used.append(scaler)
            scaled_batch = np.stack(scaled_batch, axis=0)

            # Convert to torch tensor
            x_tensor = torch.tensor(scaled_batch, dtype=torch.float32).unsqueeze(-1)  # [batch, seq_len, 1]
            dt_torch = torch.tensor(dt)

            # Compute physics loss for correct omega
            phy_loss_correct = torch.mean(
                torch.stack([
                    calculate_physics_loss(
                        x_tensor[i:i+1], 
                        omega=omega_true, 
                        dt=dt_torch, 
                        scaler=scalers_used[i]
                    )
                    for i in range(batch_size)
                ])
            ).item()

            # Compute physics loss for wrong omega
            omega_wrong = omega_true * 2.0
            phy_loss_wrong = torch.mean(
                torch.stack([
                    calculate_physics_loss(
                        x_tensor[i:i+1], 
                        omega=omega_wrong, 
                        dt=dt_torch, 
                        scaler=scalers_used[i]
                    )
                    for i in range(batch_size)
                ])
            ).item()

            print(f"[Scaler: {scaler_type}, Amp: {A}] Correct ω loss: {phy_loss_correct:.8f}, Wrong ω loss: {phy_loss_wrong:.8f}")

            # Visual check: plot residuals for wrong omega for first trajectory
            with torch.no_grad():
                x_scaled_s = x_tensor[0].squeeze(-1)
                d2x_scaled_dt2 = (x_scaled_s[2:] - 2*x_scaled_s[1:-1] + x_scaled_s[:-2]) / dt**2
                alpha = torch.tensor(scalers_used[0].scale_[0] if scaler_type=="StandardScaler" else scalers_used[0].data_max_[0]-scalers_used[0].data_min_[0])
                beta = torch.tensor(scalers_used[0].mean_[0] if scaler_type=="StandardScaler" else scalers_used[0].data_min_[0])
                x_phys_torch = alpha * x_scaled_s[1:-1] + beta
                d2x_phys_dt2 = alpha * d2x_scaled_dt2
                residual = d2x_phys_dt2 + (omega_wrong**2) * x_phys_torch

            plt.figure(figsize=(6,3))
            plt.plot(residual.numpy(), label=f"Residual (wrong ω)")
            plt.title(f"Residuals Check | Scaler: {scaler_type} | Amp: {A}")
            plt.xlabel("Time Step")
            plt.ylabel("Residual")
            plt.grid(True)
            plt.show()

            # Basic automatic check
            if phy_loss_wrong <= phy_loss_correct*10:
                print("WARNING: Physics loss not sensitive to omega. Possible unit leakage!")
            else:
                print("Physics loss responds correctly. No unit leakage detected.\n")


def test_physics_loss_units(dt, omega_true, scaler_type="StandardScaler"):
    """
    Quick test to detect unit leakage.
    Runs calculate_physics_loss on a perfect harmonic oscillator.
    """
    import torch
    from src.train import calculate_physics_loss
    
    # Simulate perfect oscillator
    T = 5.0
    t = np.arange(0, T, dt)
    x_phys = np.cos(omega_true * t)  # amplitude=1, simple harmonic
    
    # Scale the data
    if scaler_type == "StandardScaler":
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
    elif scaler_type == "MinMaxScaler":
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    else:
        raise ValueError("Unknown scaler type")
    
    x_scaled = scaler.fit_transform(x_phys.reshape(-1,1)).flatten()
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    dt_torch = torch.tensor(dt)
    
    # Physics loss with correct omega
    phy_loss_correct = calculate_physics_loss(
        reconstruction=x_tensor,
        omega=omega_true,
        dt=dt_torch,
        scaler=scaler
    )
    
    # Physics loss with wrong omega
    phy_loss_wrong = calculate_physics_loss(
        reconstruction=x_tensor,
        omega=omega_true*2.0,
        dt=dt_torch,
        scaler=scaler
    )
    
    print(f"Physics loss with correct omega ({scaler_type}): {phy_loss_correct.item():.8f}")
    print(f"Physics loss with wrong omega ({scaler_type}):   {phy_loss_wrong.item():.8f}")
    
    if phy_loss_wrong <= phy_loss_correct*10:
        print("WARNING: Physics loss is not sensitive to omega! Potential unit leakage.")
    else:
        print("Physics loss responds correctly to omega. No unit leakage detected.")


def physics_residual_physical(x_phys, omega, dt):
    d2x = (x_phys[2:] - 2*x_phys[1:-1] + x_phys[:-2]) / dt**2
    return d2x + omega**2 * x_phys[1:-1]


def run_single_model(config, X_train_tensor, train_loader, test_loader, physics_loss_weight, device, run_name, anomaly_idxs, scaler):
    """
    Runs a single training and evaluation pipeline.
    Returns: (errors_np, threshold, reconstruction_windows, detected_windows_count, detected_anomaly_tp)
    """
    print(f"\n--- Running: {run_name} (Physics Weight: {physics_loss_weight:.2f}) ---")

    # Set seed before model initialization for reproducible weight initialization
    set_all_seeds(config.RANDOM_STATE)
    
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
    device=device,
    scaler=scaler
    )
    #train_model( model, train_loader, criterion, optimizer, num_epochs, physics_loss_weight, dt, device, scaler)

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
    # SET ALL SEEDS FIRST - CRITICAL FOR REPRODUCIBILITY
    set_all_seeds(config.RANDOM_STATE)
    print("--- 1. Configuration Loaded and Data Prepared ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    
    g = torch.Generator()
    g.manual_seed(config.RANDOM_STATE)
    
    # For training data (no noise)
    x_clean = simulate_harmonic_oscillator(
        timesteps=int(config.TIMESTEPS/3),
        dt=config.DT,
        omega=config.OMEGA,
        noise_std=0.0,
        seed=config.RANDOM_STATE  # NEW
    )

    # For test data baseline (with noise)
    x_with_noise = simulate_harmonic_oscillator(
        timesteps=config.TIMESTEPS,
        dt=config.DT,
        omega=config.OMEGA,
        noise_std=0.5,
        seed=config.RANDOM_STATE + 1  # NEW - different seed for noise
    )

    print("\n--- Testing with Physics-Violating Anomalies ---")
    #x_anomalous, anomaly_idxs = inject_perturbations(
    #    x_with_noise, 
    #    num_anomalies=config.NUM_ANOMALIES, 
    #    severity=config.SEVERITY,
    #    seed=config.RANDOM_STATE + 2  # NEW
    #)
    #x_anomalous, anomaly_dict = inject_combined_physics_violations(
    #    x_with_noise,
    #    num_anomalies=config.NUM_ANOMALIES,
    #    omega_base=config.OMEGA,
    #    dt=config.DT
    #)
    #anomaly_idxs = anomaly_dict['all']

    # Visualize the violation types
    
    #import inspect
    #print(inspect.getsource(prepare_data))

    X_train_np, scaler, _ = prepare_data(x_clean, config.WINDOW_SIZE)
    #if hasattr(scaler, "scale_"):  print('--------- This is a standard scaler')
    if isinstance(scaler, StandardScaler): print("This is a StandardScaler")
    if isinstance(scaler, MinMaxScaler): print("This is a MinMaxScaler")
    X_train_tensor = torch.from_numpy(X_train_np).float().to(device)

    
    train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True,
        generator=g,  # NEW
        worker_init_fn=seed_worker  # NEW
    )

    # Anomalous data for testing (Shared by both models)
    #x_anomalous, anomaly_idxs = inject_perturbations(
    #    x_with_noise, 
    #    num_anomalies=config.NUM_ANOMALIES, 
    #    severity=config.SEVERITY
    #)

    # Choose your anomaly type:
    # Option 1: Frequency violations only
    x_anomalous, anomaly_idxs = inject_frequency_violations(
        x_with_noise,
        num_anomalies=config.NUM_ANOMALIES,
        duration=20,
        omega_base=config.OMEGA,
        dt=config.DT,
        seed=config.RANDOM_STATE + 2
    )

    # Option 2: Phase discontinuities only
    #x_anomalous, anomaly_idxs = inject_phase_discontinuities(
    #     x_with_noise,
    #     num_anomalies=config.NUM_ANOMALIES,
    #     phase_shift_range=(np.pi/4, np.pi),
    #     seed=config.RANDOM_STATE + 2
    #)

    # Option 3: Combined physics violations, needs debugging
    #x_anomalous, anomaly_dict = inject_combined_physics_violations(
    #     x_with_noise,
    #     num_anomalies=config.NUM_ANOMALIES,
    #     omega_base=config.OMEGA,
    #     dt=config.DT,
    #     seed=config.RANDOM_STATE + 2
    #
    #anomaly_idxs = anomaly_dict['all']  # Use this for existing code

    plot_physics_violation_types(
    x_with_noise, 
    x_anomalous, 
    #anomaly_dict, 
    anomaly_idxs, 
    config.WINDOW_SIZE,
    filename=os.path.join(config.RESULTS_DIR, "physics_violation_types.png")
    )

    X_test_anom_np, _, scaled_x_anomalous = prepare_data(
        x_anomalous, 
        config.WINDOW_SIZE, 
        scaler=scaler
    )
    X_test_anom_tensor = torch.from_numpy(X_test_anom_np).float().to(device)
    test_dataset = TensorDataset(X_test_anom_tensor, X_test_anom_tensor)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, worker_init_fn=seed_worker )
    
    
    # --- Run 1: PINN-Enabled Autoencoder ---
    pinn_weight = config.PHYSICS_LOSS_WEIGHT
    print(f'********* physics weight = {pinn_weight}')
    
    errors_pinn, threshold_pinn, recon_pinn_windows, detected_pinn_windows, detected_pinn_anomalies = run_single_model(
        config, X_train_tensor, train_loader, test_loader, pinn_weight, device, "PINN Autoencoder", anomaly_idxs, scaler
    )

    
    # --- Run 2: Standard Autoencoder ---
    standard_ae_weight = 0.0
    errors_standard, threshold_standard, recon_standard_windows, detected_standard_windows, detected_standard_anomalies = run_single_model(
        config, X_train_tensor, train_loader, test_loader, standard_ae_weight, device, "Standard Autoencoder", anomaly_idxs, scaler
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
    set_all_seeds(app_config.RANDOM_STATE)
    print("\n=== UNIT LEAKAGE CHECK ===")
    #test_physics_loss_units(dt=app_config.DT, omega_true=app_config.OMEGA, scaler_type="StandardScaler")
    #test_physics_loss_units(dt=app_config.DT, omega_true=app_config.OMEGA, scaler_type="MinMaxScaler")

    unit_leakage_test_suite(dt=app_config.DT, omega_true=app_config.OMEGA)
    
    print("\n=== RUNNING MAIN COMPARISON ===")
    run_comparison(app_config)
