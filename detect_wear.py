# Import libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

# Generate Synthetic Dataset
np.random.seed(42)

num_samples = 5000  # Adjust size as needed
thermal_cycles = np.random.randint(100, 5000, num_samples)
max_temperature = np.random.randint(60, 120, num_samples)
delta_temperature = np.random.randint(10, 50, num_samples)

# Simulate degradation (higher Rds_on & Vth Drift over cycles)
rds_on = 0.01 + (thermal_cycles / 5000) * 0.1 + np.random.normal(0, 0.005, num_samples)
vth_drift = 1.5 + (thermal_cycles / 5000) * 0.5 + np.random.normal(0, 0.05, num_samples)

# Wear Condition (1 = Degraded, 0 = Healthy)
wear_condition = (rds_on > 0.07).astype(int)

# Create DataFrame
data = pd.DataFrame({
    'Thermal_Cycles': thermal_cycles,
    'Max_Temperature': max_temperature,
    'Delta_Temperature': delta_temperature,
    'Rds_on': rds_on,
    'Threshold_Voltage_Drift': vth_drift,
    'Wear_Condition': wear_condition
})

print("Sample of generated data:")
print(data.head())

# Data Preprocessing
features = ['Thermal_Cycles', 'Max_Temperature', 'Delta_Temperature', 'Rds_on', 'Threshold_Voltage_Drift']
target = 'Wear_Condition'

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(data[features])
y = data[target].values

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Reshape for LSTM (batch, sequence, features)
X_tensor = X_tensor.view(X_tensor.shape[0], 1, X_tensor.shape[1])  # Add time-step dimension

# Split into 80% train, 20% test sets
train_size = int(0.8 * len(X_tensor))
X_train, X_test = X_tensor[:train_size], X_tensor[train_size:]
y_train, y_test = y_tensor[:train_size], y_tensor[train_size:]

# Create DataLoaders
batch_size = 32
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

# LSTM Model
class WearLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(WearLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()  # For binary classification

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])  # Get last time-step output
        return self.sigmoid(out)  # Output probability

# Initialize model
input_size = len(features)
hidden_size = 50
num_layers = 2
output_size = 1

model = WearLSTM(input_size, hidden_size, num_layers, output_size)

# Training Setup
criterion = nn.BCELoss()  # Binary cross-entropy loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cpu")
model.to(device)

# Train the Model
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        y_pred = model(X_batch).squeeze()
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}")

# Evaluate Model
model.eval()
y_pred_list = []

with torch.no_grad():
    for X_batch, _ in test_loader:
        X_batch = X_batch.to(device)
        y_pred = model(X_batch).squeeze()
        y_pred_list.extend(y_pred.cpu().numpy())

# Convert predictions to binary (Threshold: 0.5)
y_pred_binary = [1 if p > 0.5 else 0 for p in y_pred_list]

# Calculate Accuracy
test_accuracy = accuracy_score(y_test, y_pred_binary)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Plot Results
plt.figure(figsize=(8,5))
plt.scatter(data['Thermal_Cycles'], data['Rds_on'], c=data['Wear_Condition'], cmap='coolwarm', alpha=0.5)
plt.xlabel('Thermal Cycles')
plt.ylabel('Rds_on')
plt.title('Component Wear Distribution')
plt.colorbar(label="Wear Condition (0 = Healthy, 1 = Degraded)")
plt.show()
