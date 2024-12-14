import torch
import torch.nn as nn
import pandas as pd
from dotenv import load_dotenv
import os
from datetime import datetime
from utils import get_latest_model

here = os.path.dirname(__file__)

# Load parameters from .env
load_dotenv()

# Parameters (should match those used during training)
hidden_size = int(os.getenv("HIDDEN_SIZE"))
training_cycles = int(os.getenv("TRAINING_CYCLES"))
learning_rate = float(os.getenv("LEARNING_RATE"))
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

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and set to evaluation mode
model_file = here + "/../models/" + get_latest_model()
model = RNNInflationPredictor(input_size=input_size, hidden_size=hidden_size, output_size=output_size).to(device)
model.load_state_dict(torch.load(model_file, weights_only=True))
model.eval()

# Load Current Inflation Data
data_file = here + "/../data/datos_de_inflacion_mejorados.csv"
inflation_data = pd.read_csv(data_file)

# Extract features and prepare the latest row for prediction
features = inflation_data[["Lag_1", "Lag_3", "Lag_6", "Rolling_Mean_3", "Rolling_Std_3"]].values
latest_features = torch.tensor(features[-1], dtype=torch.float32).to(device)  # Take the last row of features
latest_features = latest_features.unsqueeze(0).unsqueeze(1)

# MAke the prediction show the results and save them in a log file
with torch.no_grad():
    prediction = model(latest_features)

log_file = here + "/../logs/predictions.log"
os.makedirs(os.path.dirname(log_file), exist_ok=True)

log_entry = f"{datetime.now().isoformat()} | Model File: {os.path.basename(model_file)} | Model Variables: (Hidden Size, Training Cycles, Learning Rate)[{hidden_size}, {training_cycles}, {learning_rate}] | Input Features: {latest_features.cpu().numpy().tolist()} --> Predicted Inflation: {prediction.item()}\n"

# Print and save the log entry
print(log_entry.strip())
with open(log_file, "a") as log:
    log.write(log_entry)