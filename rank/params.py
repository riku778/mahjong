from datetime import datetime
from pathlib import Path

INPUT_SIZE = 37
HIDDEN1_SIZE = 150
HIDDEN2_SIZE = 150
HIDDEN3_SIZE = 150
OUTPUT_SIZE = 4

DEROPOUT_RATE_INP = 0.25
DEROPOUT_RATE_HID = 0.25


EPOCHS = 300
VALIDATION_SPLIT = 0.2

LOG_DIR = Path('logs/fit') / datetime.now().strftime("%Y%m%d-%H%M%S")
MODEL_FILE_PATH = Path('model/model.h5')