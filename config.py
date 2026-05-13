import os

BASE_DIR       = r"D:\Capstone project\BUSBRA"
IMG_DIR        = os.path.join(BASE_DIR, "Images")
MASK_DIR       = os.path.join(BASE_DIR, "Masks")
CSV_PATH       = os.path.join(BASE_DIR, "bus_data.csv")
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")

ENCODER_NAME = 'timm-mobilenetv3_large_100'

IMAGE_SIZE    = (256, 256)
BATCH_SIZE    = 8
NUM_EPOCHS    = 500
LEARNING_RATE = 1e-4
NUM_WORKERS   = 2
PIN_MEMORY    = True

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.20
TEST_RATIO  = 0.10
RANDOM_SEED = 42
