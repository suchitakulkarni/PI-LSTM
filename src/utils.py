# src/utils.py
import torch
import numpy as np
import random
import os

def set_all_seeds(seed=42):
    """
    Set all random seeds for reproducibility.
    Call this at the very beginning of your script.
    
    Args:
        seed: Random seed value
    """
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # PyTorch backends
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Python hash seed (for reproducibility across runs)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"All random seeds set to: {seed}")
    print(f"CUDA deterministic mode: {torch.backends.cudnn.deterministic}")
    print(f"CUDA benchmark mode: {torch.backends.cudnn.benchmark}")


def seed_worker(worker_id):
    """
    Seed function for DataLoader workers.
    Use with DataLoader's worker_init_fn parameter.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)