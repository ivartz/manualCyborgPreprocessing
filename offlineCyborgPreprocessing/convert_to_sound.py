# Converts a csv file coming from
# Multi Channel DataManager with the export option ASCII
# to separate wav files for each electrode
# with equal (10000 Hz) sampling rate.

# Run with the command: python3 convert_to_sound.py

# Verified working on Linux Ubuntu.
# Dependencies: pip, numpy, scipy, pandas .

# <-- MEANS A PLACE YOU NEED TO EDIT.

# NB! This is a python 3 script.
# NB! These libraries need to be installed
# with python 3.
# glob and os are installed by default with python 3.
# pandas, scipy and numpy need to be installed
# with the command, using pip.

# How to install dependencies:
# Install pip (python package manager)
# then with pip, install the dependencies with
# the command:
# pip install numpy scipy pandas

# Start script.

# Import the dependencies
import glob
import pandas
from scipy.io.wavfile import write
import numpy as np
import os

# Specify the system path to (and including)
# the folder with csv files exported from
# Multi Channel DataManager with the ASCII option.
file_path_to_folder_with_raw_csv = "/media/loek/HD/Cyborg/Master thesis/testing" # <-- EDIT HERE! FOLDER WHERE csv files are placed.
                                                                                 # Full system path might be required.
# Specify the system path to the
# folder where to save the sound-
save_path_for_sound = "/media/loek/HD/Cyborg/Master thesis/testing/Sound/Fs10000" # <-- EDIT HERE! FOLDER WHERE to save sound.

# the following line stores all the csv file names as a list of strings
raw_filenames = sorted(glob.glob(file_path_to_folder_with_raw_csv + "/*.csv")) # * to select all files with file extension .csv

def find_index_of_list_containing_string(l, s):
    # takes in a list l of strings
    # returns first
    # index of the list
    # that contains the string s
    for e in l:
        if s in e:
            return l.index(e)

# SELECT the csv to import by
# writing the START OF THE FILE NAME.
experiment_index = find_index_of_list_containing_string(raw_filenames, "2") # <-- EDIT HERE! START OF THE CSV FILE NAME TO SELECT.

# Select the csv file to import.
file_to_import = raw_filenames[experiment_index]

raw_csv = True # True means skip the non-relevant lines in the beginning of the file.

if raw_csv:
    header_row = 5
elif not raw_csv:
    header_row = 0

# Load the 61 column file using pandas. This operation takes time.
print("Load csv: " + file_to_import)
raw_file = pandas.read_csv(file_to_import, sep=",", header=header_row)

# Get the header.
header = list(raw_file.head(0))

# Load, convert to sound and save electrodes.
for electrode_index in range(1,len(header)): # 1 to skip the first (TimeStamp) column

    # Load the selected column signal.
    S = np.array(raw_file[header[electrode_index]])

    # Convert and save to waveform.
    print("Convert to sound electrode: " + header[electrode_index])
    w = S.astype(np.float32)
    r = np.max(w)-np.min(w)

    # Here is the mean centering.
    wc = (w - np.mean(w))*(1/r)

    if not os.path.exists(save_path_for_sound):
        os.makedirs(save_path_for_sound)

    write(save_path_for_sound+"/"+header[electrode_index]+"_Fs_10000.wav", 10000, wc)

# End script.
