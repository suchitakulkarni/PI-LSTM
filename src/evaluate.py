# utils.py (Complete, Final Version)

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import matplotlib.pyplot as plt 
import os

def evaluate_model(model, data_loader, criterion, device):
    """
    Evaluates the model and computes the reconstruction error and reconstructed windows.
    Returns: 
    - all_errors (1D numpy array of mean-per-window errors)
    - all_reconstructed_windows (3D numpy array, N x seq_len x 1)
    """
    model.eval()
    all_errors = []
    all_reconstructed_windows = []
    with torch.no_grad():
        for data in data_loader:
            data_window = data[0].to(device)
            reconstruction = model(data_window)
            
            loss_elementwise = criterion(reconstruction, data_window) 
            error_per_window = torch.mean(loss_elementwise, dim=[1, 2])
            
            all_errors.extend(error_per_window.cpu().numpy())
            all_reconstructed_windows.append(reconstruction.cpu().numpy())
            
    all_reconstructed_windows = np.concatenate(all_reconstructed_windows, axis=0)
    return np.array(all_errors), all_reconstructed_windows
