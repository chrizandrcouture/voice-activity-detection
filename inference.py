import os
import torch
import pdb
from data import FileManager
import python_speech_features
import array
import h5py_cache


path_peer = 'data/data/'
os.path.exists(path_peer)
path_nicklas, path_simon = 'C:/Users/nickl/OneDrive/dp/retune/data', '/media/simonha/Storage/02456/data'
is_peer = os.path.exists(path_peer)
is_nicklas = os.path.exists(path_nicklas) if not is_peer else False
print('Directory is set to', os.getcwd())

# Specify the desired WAV-format.
SAMPLE_RATE = 16000
SAMPLE_CHANNELS = 1
SAMPLE_WIDTH = 2
# Name of folder to save the data files in.
DATA_FOLDER = 'data'
# Min/max length for slicing the voice files.
SLICE_MIN_MS = 1000
SLICE_MAX_MS = 5000
# Frame size to use for the labelling.
FRAME_SIZE_MS = 30
# Convert slice ms to frame size.
SLICE_MIN = int(SLICE_MIN_MS / FRAME_SIZE_MS)
SLICE_MAX = int(SLICE_MAX_MS / FRAME_SIZE_MS)
# Calculate frame size in data points.
FRAME_SIZE = int(SAMPLE_RATE * (FRAME_SIZE_MS / 1000.0))
OBJ_SHOW_PLAYABLE_TRACKS = not is_nicklas
OBJ_CUDA = torch.cuda.is_available()
OBJ_PREPARE_AUDIO = False
OBJ_TRAIN_MODELS = False
if OBJ_CUDA:
    print('CUDA has been enabled.')
else:
    print('CUDA has been disabled.')


speech_dataset = FileManager('speech', 'data')
speech_dataset.prepare_files()
speech_dataset.collect_frames()
speech_dataset.label_frames()
pdb.set_trace()

data = h5py_cache.File(DATA_FOLDER + '/data.hdf5', 'a', chunk_cache_mem_size=1024**3)
noise_levels_db = { 'None': None, '-15': -15, '-3': -3 }

mfcc_window_frame_size = 4
speech_data = speech_dataset.data
