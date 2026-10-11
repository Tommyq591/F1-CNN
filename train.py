"""PyCharm Run entry point. Configure parameters or use python -m f1_cnn train."""
from datetime import datetime
from pathlib import Path
import torch
from f1_cnn.config import TrainingConfig
from f1_cnn.pipeline import default_data_root, run_suite


if __name__ == '__main__':
    torch.set_num_threads(4)
    run_suite(default_data_root(), Path(__file__).resolve().parent / 'runs' /
              datetime.now().strftime('%Y%m%d_%H%M%S'), TrainingConfig(), folds=3)
