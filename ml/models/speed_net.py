"""
AI Speed Estimation Model (Phase 3).

Uses a CNN-GRU architecture to estimate vehicle speed from windowed IMU data.
Input: (Batch, Window_Size, 6) -> [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
Output: (Batch, 1) -> predicted speed (m/s)
"""

import torch
import torch.nn as nn

class SpeedNet(nn.Module):
    def __init__(self, input_dim=6, window_size=50, hidden_dim=64):
        super(SpeedNet, self).__init__()
        
        # 1D CNN for feature extraction from raw IMU
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=hidden_dim, kernel_size=3, padding=1)
        
        # GRU for temporal modeling
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        
        # Fully connected layer for regression
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # x shape: (Batch, Window, Channels)
        # Permute for CNN: (Batch, Channels, Window)
        x = x.permute(0, 2, 1)
        
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        
        # Permute back for GRU: (Batch, Window_reduced, Channels)
        x = x.permute(0, 2, 1)
        
        # GRU layer
        out, _ = self.gru(x)
        
        # Take the last time step
        out = out[:, -1, :]
        
        # Predict speed
        speed = self.fc(out)
        
        # Speed cannot be negative
        speed = torch.relu(speed)
        
        return speed

def create_windows(imu_data, speeds, window_size=50, stride=10):
    """Create sliding windows for training."""
    X = []
    y = []
    for i in range(0, len(imu_data) - window_size, stride):
        X.append(imu_data[i:i+window_size])
        y.append(speeds[i+window_size-1])  # Target is the speed at the end of window
    
    import numpy as np
    return np.array(X), np.array(y)
