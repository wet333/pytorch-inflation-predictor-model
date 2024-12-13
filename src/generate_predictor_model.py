import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

# Load Parameters
load_dotenv()

# Set Values
sequence_length = int(os.getenv("SEQUENCE_LENGTH"))
hidden_size = int(os.getenv("HIDDEN_SIZE"))
input_size = int(os.getenv("INPUT_SIZE"))
output_size = int(os.getenv("OUTPUT_SIZE"))
learning_rate = float(os.getenv("LEARNING_RATE"))
training_cycles = int(os.getenv("TRAINING_CYCLES"))

# Decay rate (the rate by which the gradient steps are shortened)
decay_rate = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}, {torch.cuda.get_device_name(0)}")

# Read CSV monthly inflation data
inflation_data_file = "../data/training_inflation_data.csv"
inflation_data = pd.read_csv(inflation_data_file)["Valor"].values

def create_sequences(data, seq_length = 12):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i: i + seq_length])
        targets.append(data[i + seq_length])
    return np.array(sequences), np.array(targets)

def create_variable_sequences(data):
    sequences = []
    targets = []
    for seq_length in range(len(data), 1, -1):
        sequences.append(data[:seq_length-1])
        targets.append(data[seq_length-1])
    return np.array(sequences, dtype=object), np.array(targets)

# Check if saved sequences file exists
sequences_file = "../data/sequences_data.npz"
if os.path.exists(sequences_file):
    loaded_data = np.load(sequences_file, allow_pickle=True)
    sequences = loaded_data["sequences"]
    targets = loaded_data["targets"]
    print("Sequences loaded from file: " + sequences_file + ".")
else:
    sequences, targets = create_variable_sequences(inflation_data)
    np.savez_compressed(sequences_file, sequences=sequences, targets=targets)
    print("Sequences generated and saved to file.")

sequences ,targets = create_variable_sequences(inflation_data)
sequences_tensor = [torch.tensor(seq, dtype=torch.float32).view(-1, 1) for seq in sequences]
targets_tensor = torch.tensor(targets, dtype=torch.float32).view(-1, output_size).to(device)

class RNNInflationPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNInflationPredictor, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

model = RNNInflationPredictor(input_size=input_size, hidden_size=hidden_size, output_size=output_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

print("Starting training...")

for cycle in range(training_cycles):
    model.train()
    current_loss = 0

    for seq, target in zip(sequences_tensor, targets_tensor):
        seq = seq.view(1, -1, input_size).to(device)
        target = target.view(1, -1)
        output = model(seq)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        current_loss += loss.item()

    # Decrease learning step
    if (cycle + 1) % 50 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay_rate

    print(f'Cycle {cycle + 1}/{training_cycles}, Average Loss: {current_loss}')

torch.save(model.state_dict(), "../models/inflation_predictor_model_v1.pth")