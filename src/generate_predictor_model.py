import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
from datetime import datetime
from dotenv import load_dotenv, set_key

here = os.path.dirname(__file__)

# Load NN Configuration Parameters
load_dotenv()

hidden_size = int(os.getenv("HIDDEN_SIZE"))
output_size = int(os.getenv("OUTPUT_SIZE"))
learning_rate = float(os.getenv("LEARNING_RATE"))
training_cycles = int(os.getenv("TRAINING_CYCLES"))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}, {torch.cuda.get_device_name(0)}")

# Read CSV monthly inflation data
inflation_data_file = here + "/../data/datos_de_inflacion_mejorados.csv"
inflation_data = pd.read_csv(inflation_data_file)

features = inflation_data[["Lag_1", "Lag_3", "Lag_6", "Rolling_Mean_3", "Rolling_Std_3"]].values
targets = inflation_data["Valor"].values

features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
targets_tensor = torch.tensor(targets, dtype=torch.float32).view(-1, 1).to(device)

input_size = features.shape[1]

# Save input_size to .env for running the model later
env_file = here + "/../.env"
set_key(env_file, "INPUT_SIZE", str(input_size))

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
    total_loss = 0

    for feature, target in zip(features_tensor, targets_tensor):
        # Reshape feature for model input
        feature = feature.unsqueeze(0).unsqueeze(1).to(device)
        target = target.unsqueeze(0).to(device)

        # Forward pass
        output = model(feature)
        loss = criterion(output, target)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f'Cycle {cycle + 1}/{training_cycles}, Loss: {total_loss / len(features_tensor):.4f}')

torch.save(model.state_dict(), here + "/../models/inflation_predictor_model_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".pth")