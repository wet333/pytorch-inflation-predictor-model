import os
import re
from datetime import datetime

here = os.path.dirname(__file__)

def get_latest_model():
    """
    Finds the latest model file generated in the "models" directory based on the timestamp in the filename.

    :return: The latest model's filename
    """
    models_dir = os.path.join(here, "../models")
    model_pattern = re.compile(r"inflation_predictor_model_(\d{8}_\d{6})\.pth")
    latest_model = None
    latest_timestamp = None

    for file_name in os.listdir(models_dir):
        match = model_pattern.match(file_name)
        if match:
            timestamp_str = match.group(1)
            timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            if latest_timestamp is None or timestamp > latest_timestamp:
                latest_timestamp = timestamp
                latest_model = os.path.join(models_dir, file_name)

    return os.path.basename(latest_model)