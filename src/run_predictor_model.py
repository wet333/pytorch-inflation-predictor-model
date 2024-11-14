import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os

# Load parameters from .env
load_dotenv()

# Parameters (should match those used during training)
sequence_length = int(os.getenv("SEQUENCE_LENGTH"))
hidden_size = int(os.getenv("HIDDEN_SIZE"))
input_size = int(os.getenv("INPUT_SIZE"))
output_size = int(os.getenv("OUTPUT_SIZE"))

# Define model class
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and set to evaluation mode
model = RNNInflationPredictor(input_size=input_size, hidden_size=hidden_size, output_size=output_size).to(device)
model.load_state_dict(torch.load("../models/inflation_predictor_model_v1.pth", weights_only=True))
model.eval()

def create_sequences(data, seq_length=12):
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i: i + seq_length])
    return np.array(sequences)

# Load new data
data_file = "../data/prediction_inflation_data.csv"
inflation_data = pd.read_csv(data_file)["Valor"].values

# Prepare input sequence from the end of new data
sequences = create_sequences(inflation_data, sequence_length)
last_sequence = torch.tensor(sequences[-1], dtype=torch.float32).view(1, sequence_length, input_size).to(device)

# Make prediction
with torch.no_grad():
    prediction = model(last_sequence)
    print(f'{sequences[-1]} --> {prediction.item()}')
