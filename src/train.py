# utils.py (Complete, Final Version)

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import matplotlib.pyplot as plt 
import os


def calculate_physics_loss(reconstruction, omega, dt):
    """Calculates the Physics-Informed Loss (MSE of the residual)."""
    x = reconstruction.squeeze(-1) 
    d2x_dt2 = (x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]) / (dt ** 2)
    x_central = x[:, 1:-1]
    F_residual = d2x_dt2 + (omega ** 2) * x_central
    loss_physics = torch.mean(F_residual ** 2)
    return loss_physics

def train_model(model, train_loader, criterion, optimizer, num_epochs, 
                physics_loss_weight, dt, device):

    model.train()
    history = {'mse_loss': [], 'physics_loss': [], 'total_loss': []}

    for epoch in range(1, num_epochs + 1):
        epoch_total = 0.0
        epoch_mse = 0.0
        epoch_phy = 0.0

        for data in train_loader:
            data_window = data[0].to(device)
            optimizer.zero_grad()

            reconstruction = model(data_window)

            # Reconstruction loss (MSE)
            mse_loss = torch.mean(criterion(reconstruction, data_window))

            # Physics loss
            if physics_loss_weight > 0:
                phy_loss = calculate_physics_loss(
                    reconstruction=reconstruction,
                    omega=2.0,
                    dt=dt
                )
            else:
                phy_loss = 0.0

            # Combined loss
            total_loss = mse_loss + physics_loss_weight * phy_loss

            # Backprop
            total_loss.backward()
            optimizer.step()

            # Accumulate epoch totals (as scalars)
            epoch_total += total_loss.item()
            epoch_mse += mse_loss.item()
            epoch_phy += (phy_loss.item()  * physics_loss_weight) if not isinstance(phy_loss, float) else phy_loss

        # Average per epoch
        n = len(train_loader)
        history['mse_loss'].append(epoch_mse / n)
        history['physics_loss'].append(epoch_phy / n)
        history['total_loss'].append(epoch_total / n)

        if epoch == num_epochs:
            print(f"Epoch {epoch:03d}/{num_epochs} | Loss: {epoch_total/n:.6f}")

    return model, history