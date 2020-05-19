import numpy as np
import os
from keras.layers import Dense, Activation, Flatten, Dropout, Concatenate, BatchNormalization, Convolution1D, MaxPooling1D
from keras.optimizers import Adam
from keras import Input, Model
import h5py
import tensorflow as tf
from keras import backend as K
import matplotlib as mpl
mpl.use('Agg')
import argparse
threads = 30
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score  ###計算roc和auc

parser = argparse.ArgumentParser(
    description="""Take a raw sequence and a labels bed like file and encode and store
    both as numpy arrays. Split up into traiing, test and validation samples.""")
parser.add_argument('--in_file',
                    help='Six column file. [<chr> <start> <end> <comma separated IDs> <raw sequence> <epi mark>].',
                    required=True)
parser.add_argument('--out_name',
                    help='Output file destination name, suffix',
                    required=True)
parser.add_argument('--predict_name',
                    help='Output prediction file name, prefix',
                    required=True)
parser.add_argument('--data_selector',
                    help='select from "testing" or "training"',
                    required=True)
parser.add_argument('--in_model',
                    help='Output file destination name, suffix',
                    required=True)
parser.add_argument('--threshold',
                    help='Output file destination name, suffix',
                    default=0.5)
args = parser.parse_args()

""" Defining initial parameters """
out_name = args.out_name
predict_name = args.predict_name
data_selector = args.data_selector
num_class = 1


def recall_keras(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_keras(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_keras(y_true, y_pred):
    precision = precision_keras(y_true, y_pred)
    recall = recall_keras(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def recall_m(y_true, y_pred):
    y_true = K.round(y_true)
    y_pred = K.round(y_pred)
    true_positive_count = K.sum(K.cast(K.equal(tf.add(y_true, y_pred), 2.), K.floatx()))
    label_positive_count = K.sum(y_true)
    recall = true_positive_count / label_positive_count
    return recall


def accuracy_m(y_true, y_pred):
    y_true = K.round(y_true)
    y_pred = K.round(y_pred)
    correct_predictions = K.sum(K.cast(K.equal(y_true, y_pred), K.floatx()))
    all_instances = K.sum(K.cast(K.greater_equal(y_true, 0.), K.floatx()))
    accuracy = correct_predictions / all_instances
    return accuracy


def precision_m(y_true, y_pred):
    y_true = K.round(y_true)
    y_pred = K.round(y_pred)
    true_positive_count = K.sum(K.cast(K.equal(tf.add(y_true, y_pred), 2.), K.floatx()))
    pred_positive_count = K.sum(y_pred)
    precision = true_positive_count / pred_positive_count
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall))


def multi_category_focal_loss2_fixed(y_true, y_pred):
    epsilon = 1.e-7
    gamma=2.
    alpha_v = .7
    alpha = tf.constant([alpha_v], dtype=tf.float32)

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

    alpha_t = y_true*alpha + (tf.ones_like(y_true)-y_true)*(1-alpha)
    y_t = tf.multiply(y_true, y_pred) + tf.multiply(1-y_true, 1-y_pred)
    ce = -tf.log(y_t)
    weight = tf.pow(tf.subtract(1., y_t), gamma)
    fl = tf.multiply(tf.multiply(weight, ce), alpha_t)
    loss = tf.reduce_mean(fl)

    return loss

train_file = args.in_file
# ----- Start to Load data! -----
# Get the sets of seqs and labels from hdf5 formated file
print("Start to read from h5 files to Numpy array:")
h5f = h5py.File(train_file, 'r')
tr_total_len = len(h5f['training_seqs'])
te_total_len = len(h5f['test_seqs'])
epi_feature_count = len(h5f['training_epi_marks'][0])

dna_train = h5f['training_seqs'][0:tr_total_len]
y_train = h5f['training_labels'][0:tr_total_len]
epi_train = h5f['training_epi_marks'][0:tr_total_len].reshape([tr_total_len, epi_feature_count, 1])


dna_test = h5f['test_seqs'][0:te_total_len]
y_test = h5f['test_labels'][0:te_total_len]
epi_test = h5f['test_epi_marks'][0:te_total_len].reshape([te_total_len, epi_feature_count, 1])
print("Finished reading training....")

print('\n\ntraining_data')
print(np.shape(dna_train))
training_cases = np.shape(dna_train)[0]
print('test_data')
print(dna_test.shape)
print('\n\n')
test_cases = np.shape(dna_train)[0]

in_model = args.in_model

print("Loading model from existing h5 file....")
model = tf.contrib.keras.models.load_model(in_model, custom_objects={
    'accuracy_m': accuracy_m, 'recall_m': recall_m, 'precision_m': precision_m, 'f1_m': f1_m,
    'recall_keras': recall_keras, 'precision_keras': precision_keras, 'f1_keras': f1_keras})
print('\nOutput checkpoint testing result------------')

# ----- Start training!! -----
if not os.path.exists('../reports/%s' % out_name):
    os.makedirs('../reports/%s' % out_name)

print('\n---- Testing ------------')
print('\nOutput checkpoint testing result------------')
if data_selector == "training":
    dna_test = dna_train
    epi_test = epi_train
    y_test = y_train
predicts = model.predict([dna_test, epi_test])
true_positives = 0
predicted_positives = 0
truth_positives = 0
correct_predictions = 0
total_negatives = 0
true_negatives = 0
threshold = args.threshold
output_predict_fh = open('../reports/%s/%s_predict.out' % (out_name, predict_name), 'w')
for x, y in zip(predicts, y_test):
    output_predict_fh.write("%s\t%s\n" % (x[0], y[0]))
    if x[0] > threshold and y[0] > threshold:
        true_positives += 1
        correct_predictions += 1
    if x[0] > threshold:
        predicted_positives += 1
    if y[0] > threshold:
        truth_positives += 1
    if x[0] < threshold and y[0] < threshold:
        correct_predictions += 1
    if y[0] < threshold:
        total_negatives += 1
    if x[0] < threshold:
        true_negatives += 1
recall = true_positives / truth_positives
precision = true_positives / predicted_positives
f1 = 2 * recall * precision / (recall + precision)
accuracy = correct_predictions / len(predicts)
specificity = true_negatives / total_negatives

print('test loss: ----')
print('test accuracy: ', accuracy)
print('test recall: ', recall)
print('test specificity: ', specificity)
print('test precision: ', precision)
print('test f1_score: ', f1)
print('test average precision score: %d' % average_precision_score(y_test, predicts))
print('test : %d' % average_precision_score(y_test, predicts))
print('test truth_positives %s' % str(truth_positives))
print('test true_positives %s' % str(true_positives))
print("%f\t%f\t%f\t%f\n" % (accuracy, recall, precision, f1))
fpr, tpr, threshold = roc_curve(y_test, predicts)  ###計算真正率和假正率

output_predict_fh.write("PREDICTED_RESULT\taccuracy\trecall\tprecision\tf1\n")
output_predict_fh.write("PREDICTED_RESULT\t%f\t%f\t%f\t%f\n" % (accuracy, recall, precision, f1))
output_predict_fh.close()
