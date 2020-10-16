"""
epimark的training data一定是singel class.
Convert the raw sequence and the lables to hdf5 data/arrays for faster batch reading.
Split data into training, test and validation set. Save training and test set in same file.
Will store a .h5 file with the labels and sequences and a coord file per test/valid and train set
"""
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import numpy as np
import h5py
import argparse
from operator import itemgetter

# Define arguments -------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="""Take a raw sequence and a labels bed like file and encode and store
    both as numpy arrays. Split up into traiing, test and validation samples.""")
parser.add_argument('--data_prefix', type=str,
                    help='Cell type to indicate multiple column file data. [<chr> <start> <end> <label> <raw sequence> <epi marks>].')
# Parse arguments
args = parser.parse_args()
data_prefix = args.data_prefix

# Helper get hotcoded sequence
def get_hot_coded_seq(sequence):
    """Convert a 4 base letter sequence to 4-row x-cols hot coded sequence"""
    # initialise empty
    hotsequence = np.zeros((len(sequence),4))
    # set hot code 1 according to gathered sequence
    for i in range(len(sequence)):
        if sequence[i] in ['A','a']: #== 'A':
            hotsequence[i,0] = 1
        elif sequence[i] in ['C','c']: # == 'C':
            hotsequence[i,1] = 1
        elif sequence[i] in ['G','g']:  # == 'G':
            hotsequence[i,2] = 1
        elif sequence[i] in ['T','t']: # == 'T':
            hotsequence[i,3] = 1
    # return the numpy array
    return hotsequence

print("\n# === Creating a Training, Test and Validation Set from provided input === #")

# Read in data -----------------------------------------------------------------
print("\nReading lines ...")

# Process testing data (k562 sample)
with open("%s_dnase.test.dat" % args.data_prefix, "r") as f:
    chroms_te = []
    start_te = []
    stop_te = []
    label_te = []
    label_tmp_te = []
    epi_mark_te = []
    # seq = []
    for i,l in enumerate(f):
        l = l.rstrip()
        l = l.split("\t")
        chroms_te.append(l[0])
        start_te.append(l[1])
        stop_te.append(l[2])
        label_te.append(int(l[3]))
        epi_length = len(l[5:])
        epi_mark_te.append([float(m) for m in l[5:]])
        # get first sequence to estimate length and format
        if i == 0:
            temp_seq = l[4]
            # trim if desired
            temp_seq = get_hot_coded_seq(temp_seq)

# print a table with the intial labels for future reference
print("\nConverting to binary representation:")
# Go through labels per seq and sum up a binary representing all active IDs ----
label_bin_te = np.zeros((len(label_te), 1),  dtype=np.int)
for j in range(len(label_te)):
    label_bin_te[j,] = label_bin_te[j,] + label_te[j]

epi_bin_te = np.zeros((len(epi_mark_te), epi_length),  dtype=np.float)
for j in range(len(epi_mark_te)):
    epi_bin_te[j,] = epi_mark_te[j]

# Sample Test/ Validation and Training set according to selected mode -----------
test_rows = np.array(range(len(chroms_te)))  # make an array of input rows to sample from once


print("\nSampled into sets ...")

# write training/test/validation set coords ------------------------------------
print("\nStoring Coordinates ...")
write_test_coords = open("%s_dnase_dataset_test_coords.bed" % args.data_prefix, "w")
for tr in test_rows:
    write_test_coords.write("%s\t%s\t%s\n" % (chroms_te[tr], start_te[tr], stop_te[tr]))

# Initialize training and validation data in hdf5 files ------------------------
print("\nInitializing hdf5 Storage Files ...")
# and already store labels
train_h5f = h5py.File("%s_dnase_dataset.h5" % args.data_prefix, 'w')

set_test_seq = train_h5f.create_dataset('test_seqs', (test_rows.shape[0], temp_seq.shape[0], temp_seq.shape[1]), dtype='i')
train_h5f.create_dataset('test_labels', data=label_bin_te[test_rows,])
train_h5f.create_dataset('test_epi_marks', data=epi_bin_te[test_rows,])

one_hot_seq = {}
print("\nRunning through raw file again, converting sequences and store in sets ...")

with open("%s_dnase.test.dat" % args.data_prefix, "r") as f:
    seq = []
    # make iterators
    test_i = 0
    valid_i = 0
    train_i = 0
    skip_count = 0
    for i,l in enumerate(f):
        l = l.rstrip()
        l = l.split("\t")
        # get sequence
        seq = l[4]
        # get first sequence length
        if i == 0:
            seq_length = len(seq)
            print("Converting and storing sequences of length %s bp." % (seq_length))
        # check sequence length matches
        if len(seq) < seq_length:
            # skip otherwise
            skip_count = skip_count + 1
            continue
        # convert to one hot coded
        if seq in one_hot_seq:
            seq = one_hot_seq[seq]
        else:
            one_hot_seq[seq] = get_hot_coded_seq(seq)
            seq = one_hot_seq[seq]
        # match and write to respective hdf5 file
        if i in test_rows[:]:
            set_test_seq[test_i,] = seq
            test_i += 1
        if i % 10000 == 0:
            print('Written lines ... %s' % (i))
    print("Skipped %s elements with sequence length != %s" % (skip_count, seq_length))

# Close
train_h5f.close()

print("\nSaved the data Data.\n")